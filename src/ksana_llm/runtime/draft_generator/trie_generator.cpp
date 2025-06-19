/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/runtime/draft_generator/trie_generator.h"

namespace ksana_llm {

void TrieTree::Put(const std::vector<int>& tokens, size_t begin, size_t end, int is_final, FrequencyType type,
                   int pos_id) {
  Node* current = root_;
  for (size_t i = begin; i < end; ++i) {
    const auto token = tokens[i];
    const auto& [node_it, is_new] = current->children.try_emplace(token, new Node(token));
    if (is_new) {
      ++current_nodes_;
    }
    current = node_it->second;
    if (type == FrequencyType::INPUT) {
      current->input_freqs[pos_id] += 1.0;
    } else {
      current->output_freq += 1.0;
    }
  }
  Prune();
}

void TrieTree::DFSCollectFreq(const Node* node, std::vector<double>& results, int pos_id, double output_weight) const {
  const double user_freq = node->input_freqs.count(pos_id) ? node->input_freqs.at(pos_id) : 0.0;
  const double score = user_freq * (1.0 - output_weight) + node->output_freq * output_weight;
  results.emplace_back(score);

  for (const auto& entry : node->children) {
    DFSCollectFreq(entry.second, results, pos_id, output_weight);
  }
}

void TrieTree::QueryOneBranch(const std::vector<int>& token_ids, size_t begin, size_t end, int pos_id,
                              std::vector<int>& result_ids, size_t max_depth, double output_weight) const {
  result_ids.clear();
  Node* current = root_;
  if (max_depth == 0) return;

  // Traverse the trie
  for (size_t i = begin; i < end; ++i) {
    const auto token = token_ids[i];
    auto it = current->children.find(token);
    if (it == current->children.end()) return;
    current = it->second;
  }

  // Find the optimal branch
  while (current != nullptr && result_ids.size() < max_depth) {
    Node* best_child = nullptr;
    double max_score = -1.0;

    for (const auto& entry : current->children) {
      double input_freq = entry.second->input_freqs.count(pos_id) ? entry.second->input_freqs.at(pos_id) : 0.0;
      double score = input_freq * (1.0 - output_weight) + entry.second->output_freq * output_weight;

      if (score > max_score) {
        max_score = score;
        best_child = entry.second;
      }
    }

    if (best_child) {
      result_ids.push_back(best_child->token_id);
      current = best_child;
    } else {
      break;
    }
  }
}

void TrieTree::Reset() {
  DFSDelete(root_, 1);
  root_ = new Node(0);
  current_nodes_ = 1;
}

void TrieTree::ResetInputFreqs(int pos_id) { ResetInputFreqsHelper(root_, pos_id); }

void TrieTree::ResetInputFreqsHelper(Node* node, int pos_id) {
  if (node->input_freqs.count(pos_id)) {
    node->input_freqs.erase(pos_id);
  }
  for (auto& entry : node->children) {
    ResetInputFreqsHelper(entry.second, pos_id);
  }
}

bool TrieTree::DFSDelete(Node* node, size_t target_node_num) {
  if (!node || current_nodes_ <= target_node_num) return false;

  std::vector<int> to_delete;
  for (auto& entry : node->children) {
    if (DFSDelete(entry.second, target_node_num)) {
      to_delete.push_back(entry.first);
    }
  }

  for (int token : to_delete) {
    node->children.erase(token);
  }

  if (node != root_ && node->children.empty() && current_nodes_ > target_node_num) {
    delete node;
    current_nodes_--;
    return true;
  }
  return false;
}

void TrieTree::Prune() {
  if (current_nodes_ <= max_nodes_) {
    return;
  }
  DFSDelete(root_, max_nodes_);
}

void TrieGenerator::GenerateDraft(const std::vector<int>& input_tokens, int step, int suggested_draft_num,
                                  std::vector<int>& draft_tokens, int unverfied_token_num, int accepted_tokens,
                                  int req_id) {
  KLLM_CHECK(input_tokens.size() >= unverfied_token_num);
  const size_t usable_length = input_tokens.size() - unverfied_token_num;  // exclude mtp tokens
  const size_t update_begin =
      step <= 1 ? 0 : usable_length - std::min(kStepGenerateTokenNum + accepted_tokens + kMaxDepth - 1, usable_length);
  thread_local std::vector<int> tokens;  // TODO(lijiajieli): avoid copy
  KLLM_CHECK(input_tokens.size() >= update_begin);
  tokens.assign(input_tokens.begin() + update_begin, input_tokens.begin() + usable_length);
  const TrieTree::FrequencyType type = step <= 1 ? TrieTree::FrequencyType::INPUT : TrieTree::FrequencyType::OUTPUT;
  // TODO(qiannan): Need to determine if 'final' is appropriate for modifying the tree.
  StreamPut(tokens, 0, false, type, req_id);

  // TODO(qiannan): kOutputWeight should be dynamically adjusted according to the length of the input request.
  constexpr double kOutputWeight = 0.2;
  tokens.insert(tokens.end(), input_tokens.begin() + usable_length, input_tokens.end());
  Predict(tokens, draft_tokens, suggested_draft_num, 0, "OneBranch", kOutputWeight);
}

void TrieGenerator::StreamPut(const std::vector<int>& token_ids, int last_pos, bool is_final,
                              TrieTree::FrequencyType type, int pos_id) {
  while (last_pos + kMaxDepth <= token_ids.size()) {
    const int end_pos = last_pos + kMaxDepth;
    trie_->Put(token_ids, last_pos, end_pos, is_final, type, pos_id);
    last_pos++;
  }
  if (is_final) {
    trie_->Prune();
    trie_->ResetInputFreqs(pos_id);
  }
}

void TrieGenerator::Predict(const std::vector<int>& token_ids, std::vector<int>& spec_ids, size_t max_output_size,
                            int pos_id, const std::string& function_name, double output_weight) {
  if (token_ids.empty() || max_output_size == 0) {
    return;
  }

  const size_t len = token_ids.size();
  for (size_t i = len - std::min(kMaxMatchDepth, len); i < len; ++i) {
    size_t max_draft_num = std::min(max_output_size, kMaxDepth - kMaxMatchDepth);
    max_draft_num = std::min(max_draft_num, (len - i) * kPredictNumPerMatch);
    trie_->QueryOneBranch(token_ids, i, len, pos_id, spec_ids, max_draft_num, output_weight);
    if (!spec_ids.empty()) {
      return;
    }
  }
}

}  // namespace ksana_llm