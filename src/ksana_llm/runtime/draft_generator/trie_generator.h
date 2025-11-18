/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <vector>
#include "ksana_llm/runtime/draft_generator/draft_generator_interface.h"

namespace ksana_llm {

class TrieTree {
 private:
  struct Node {
    int token_id;
    std::unordered_map<int, Node*> children;
    std::unordered_map<int, double> input_freqs;
    double output_freq;

    inline explicit Node(int id) : token_id(id), output_freq(0.0) {}
  };

  size_t max_nodes_;
  size_t current_nodes_;
  Node* root_;

 public:
  friend class TrieTreeFriendTest;
  enum class FrequencyType { INPUT, OUTPUT };
  inline explicit TrieTree(int max_nodes) : max_nodes_(max_nodes), current_nodes_(1), root_(new Node(0)) {}
  TrieTree() : max_nodes_(1 << 25), current_nodes_(1), root_(new Node(0)) {}
  ~TrieTree() { Reset(); }

  void ResetInputFreqsHelper(Node* node, int pos_id);
  bool DFSDelete(Node* node, size_t target_node_num);

  Node* GetRoot() const { return root_; }
  void Put(const std::vector<int>& tokens, size_t begin, size_t end, int final, FrequencyType type, int pos_id);
  void DFSCollectFreq(const Node* node, std::vector<double>& results, int pos_id, double output_weight) const;
  void QueryOneBranch(const std::vector<int>& token_ids, size_t begin, size_t end, int pos_id,
                      std::vector<int>& result_ids, size_t max_depth, double output_weight) const;
  size_t GetCurrentNodes() const { return current_nodes_; }
  void ResetInputFreqs(int pos_id);
  void Prune();
  void Reset();
};

class TrieGenerator : public DraftGeneratorInterface {
 public:
  TrieGenerator() : trie_(new TrieTree()) {}
  inline explicit TrieGenerator(int max_nodes) : trie_(new TrieTree(max_nodes)) {}
  ~TrieGenerator() { delete trie_; }

  void GenerateDraft(std::shared_ptr<InferRequest> req) override;

 private:
  void StreamPut(const std::vector<int>& token_ids, int last_pos, bool final, TrieTree::FrequencyType type, int pos_id);
  void Predict(const std::vector<int>& token_ids, std::vector<int>& spec_ids, size_t max_output_size, int pos_id,
               const std::string& function_name, double output_weight);
  TrieTree* GetTrie() { return trie_; }

 private:
  static constexpr size_t kMaxDepth = 16;
  static constexpr size_t kMaxMatchDepth = 4;
  static constexpr size_t kPredictNumPerMatch = 3;  // match 1 token can predict 3 token
  TrieTree* trie_;
};

}  // namespace ksana_llm