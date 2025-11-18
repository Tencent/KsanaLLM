/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/draft_generator/trie_generator.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class TrieTreeFriendTest : public TrieTree {
 public:
  using TrieTree::TrieTree;
  size_t GetMaxNodes() const { return max_nodes_; }

  void Display(const Node* root) const {
    if (!root) {
      std::cout << "Tree is empty." << std::endl;
      return;
    }
    std::cout << "Trie Tree Structure:" << std::endl;
    DisplayHelper(root, 0);
  }

  void DisplayHelper(const Node* node, int indent) const {
    // First level indentation
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Token: " << node->token_id << std::endl;

    // Display input frequencies
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "User Frequencies:" << std::endl;

    std::vector<std::pair<int, double>> sorted_freqs(node->input_freqs.begin(), node->input_freqs.end());
    std::sort(sorted_freqs.begin(), sorted_freqs.end());

    for (const auto& [key, freq] : sorted_freqs) {
      for (int i = 0; i < indent; ++i) std::cout << "  ";
      std::cout << "  " << key << ": " << std::fixed << std::setprecision(4) << freq << std::endl;
    }

    // Display output frequency
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Output Frequency: " << std::fixed << std::setprecision(4) << node->output_freq << std::endl;

    // Recursively display children nodes
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Children:" << std::endl;
    if (node->children.empty()) {
      for (int i = 0; i < indent; ++i) std::cout << "  ";
      std::cout << "  None" << std::endl;
    } else {
      for (const auto& [token, child] : node->children) {
        DisplayHelper(child, indent + 2);
      }
    }
  }
};

class TrieTreeTest : public testing::Test {
 protected:
  void SetUp() override { trie_ = std::make_shared<TrieTreeFriendTest>(); }

  void TearDown() override { trie_->Reset(); }

  std::shared_ptr<TrieTreeFriendTest> trie_;
};

TEST_F(TrieTreeTest, PutBasic) {
  std::vector<int> tokens = {1, 2, 3};
  trie_->Put(tokens, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  EXPECT_EQ(trie_->GetCurrentNodes(), 4);
  auto node = trie_->GetRoot();
  for (int token : tokens) {
    auto it = node->children.find(token);
    ASSERT_NE(it, node->children.end());
    EXPECT_DOUBLE_EQ(it->second->input_freqs.at(0), 1.0);
    node = it->second;
  }
}

TEST_F(TrieTreeTest, QueryOneBranchWithThreshold) {
  // Prepare test data
  const std::vector<int> path1 = {1, 2, 3};
  const std::vector<int> path2 = {1, 2, 4};
  trie_->Put(path1, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(path2, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute query
  const std::vector<int> token_ids = {1, 2};
  std::vector<int> result_ids;
  trie_->QueryOneBranch(token_ids, 0, 2, 0, result_ids, 10, 1.0);

  // Verify results
  ASSERT_EQ(result_ids.size(), 1);
  EXPECT_TRUE(result_ids[0] == 3 || result_ids[0] == 4);
}

TEST_F(TrieTreeTest, DFSDelete) {
  // Prepare test data
  const std::vector<int> path1 = {1, 2, 3};
  const std::vector<int> path2 = {1, 2, 4};
  trie_->Put(path1, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(path2, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute deletion
  trie_->DFSDelete(trie_->GetRoot(), 1);
  EXPECT_EQ(trie_->GetCurrentNodes(), 1);
}

TEST_F(TrieTreeTest, ResetInputFreqs) {
  // Prepare test data
  const std::vector<int> tokens = {1, 2};
  trie_->Put(tokens, 0, 2, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(tokens, 0, 2, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute reset
  trie_->ResetInputFreqs(0);

  // Verify reset results
  auto node = trie_->GetRoot()->children.at(1)->children.at(2);
  EXPECT_EQ(node->input_freqs.count(0), 0);
}

class TrieGeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    generator_ = std::make_shared<TrieGenerator>();

    auto ksana_python_input = std::make_shared<KsanaPythonInput>();
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
    request_ = std::make_shared<InferRequest>(request, 1);
  }

  void TearDown() override { generator_->GetTrie()->Reset(); }

  std::shared_ptr<TrieGenerator> generator_;
  std::shared_ptr<InferRequest> request_;
};

TEST_F(TrieGeneratorTest, StreamPutAndPredict) {
  // Prepare test data
  std::vector<int> tokens1;
  for (int i = 0; i < 50; ++i) tokens1.push_back(i);

  std::vector<int> tokens2;
  for (int i = 2; i <= 50; i += 2) tokens2.push_back(i);
  // Execute operations
  generator_->StreamPut(tokens1, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  generator_->StreamPut(tokens2, 0, false, TrieTree::FrequencyType::OUTPUT, 0);

  std::vector<int> spec_ids;
  generator_->Predict({1001, 1003, 1005, 2}, spec_ids, 10, 0, "Hierarchy", 1.0);

  // Verify results
  EXPECT_EQ(spec_ids.size(), 3);
}

TEST_F(TrieGeneratorTest, PredictOneBranchSuccess) {
  std::vector<int> tokens1;
  for (int i = 2; i <= 50; i++) tokens1.emplace_back(i);
  std::vector<int> tokens2;
  for (int i = 2; i <= 50; i += 2) tokens2.emplace_back(i);
  generator_->StreamPut(tokens1, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  generator_->StreamPut(tokens2, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  std::vector<int> token_ids = {1, 2, 3, 4, 5, 6};
  std::vector<int> spec_ids;
  generator_->Predict(token_ids, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_EQ(spec_ids.size(), 10);
  EXPECT_EQ(spec_ids[0], 7);
}

TEST_F(TrieGeneratorTest, PredictEmpty) {
  std::vector<int> tokens1;
  for (int i = 0; i <= 10; i++) {
    tokens1.emplace_back(i);
  }
  std::vector<int> token_ids = {99999};
  std::vector<int> spec_ids;
  generator_->Predict(token_ids, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());

  generator_->Predict({}, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());

  generator_->Predict({}, spec_ids, 0, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());
}

TEST_F(TrieGeneratorTest, BasicTest) {
  // std::shared_ptr<InferRequest> request_;
  request_->step = 1;
  request_->suggested_draft_num = 16;
  request_->forwarding_tokens_draft_num = 0;
  request_->forwarding_tokens = {1001, 1003, 1005, 10, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1005, 10};
  request_->accepted_tokens = {};
  request_->generated_tokens = {2};
  request_->draft_tokens.mtp = {};

  generator_->GenerateDraft(request_);
  std::vector<int> draft_tokens = request_->draft_tokens.GetDraftTokens();
  EXPECT_EQ(9, draft_tokens.size());
  EXPECT_EQ(3, draft_tokens[0]);
}

TEST_F(TrieGeneratorTest, HitRateTest) {
  const std::string csv_path = "speculative_decoding_accuracy.csv";
  int total_hits = 0;
  int total_tokens = 0;
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    GTEST_SKIP() << "Skipping test: Missing required CSV file " << csv_path;
    return;
  }
  std::string line;
  std::getline(file, line);
  int req = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string prefix_str, response_str;
    std::getline(iss, prefix_str, ',');
    std::getline(iss, response_str);
    auto parse_tokens = [](const std::string& str) {
      std::vector<int> tokens;
      std::istringstream token_stream(str);
      int token;
      while (token_stream >> token) {
        tokens.push_back(token);
      }
      return tokens;
    };
    const auto prefix = parse_tokens(prefix_str);
    const auto response = parse_tokens(response_str);
    // Simulate incremental decoding process
    request_->forwarding_tokens = prefix;
    size_t pos = 0;
    request_->step = 1;
    request_->accepted_tokens.clear();
    request_->suggested_draft_num = 8;
    while (pos < response.size()) {
      generator_->GenerateDraft(request_);
      std::vector<int> draft_tokens = request_->draft_tokens.GetDraftTokens();
      request_->step += 1;
      total_tokens += draft_tokens.size();
      int accepted_token = 0;

      for (size_t i = 0; i < draft_tokens.size(); i++) {
        if (pos < response.size() && draft_tokens[i] == response[pos]) {
          request_->forwarding_tokens.push_back(response[pos]);
          ++total_hits;
          ++pos;
          ++accepted_token;
        } else {
          break;
        }
      }
      std::cout << " draft_tokens.size() = " << draft_tokens.size() << " accepted_token = " << accepted_token
                << std::endl;
      if (pos < response.size()) {
        request_->forwarding_tokens.push_back(response[pos]);
        pos++;
      }
    }
    const double actual_hit_rate = total_tokens > 0 ? static_cast<double>(total_hits) / total_tokens : 0.0;
    std::cout << "Hit tokens: " << total_hits << "\n";
    std::cout << "Total tokens: " << total_tokens << "\n";
    std::cout << "Hit rate: " << actual_hit_rate * 100 << "%\n";
    ++req;
  }
}

}  // namespace ksana_llm