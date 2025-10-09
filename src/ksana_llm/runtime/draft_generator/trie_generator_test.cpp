/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/draft_generator/trie_generator.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "ksana_llm/batch_scheduler/structured_generation/structured_generator_interface.h"
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
  void SetUp() override { generator_ = std::make_shared<TrieGenerator>(); }

  void TearDown() override { generator_->GetTrie()->Reset(); }

  std::shared_ptr<TrieGenerator> generator_;
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
  std::vector<int> input_tokens = {1001, 1003, 1005, 10, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1005, 10, 2};
  int step = 1;
  int suggested_draft_num = 16;
  std::vector<int> draft_tokens;

  generator_->GenerateDraft(input_tokens, step, suggested_draft_num, draft_tokens, 0, 0, 1);

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
    std::vector<int> input_tokens = prefix;
    size_t pos = 0;
    int step = 1, accepted_token = 0;
    while (pos < response.size()) {
      std::vector<int> draft_tokens;
      const int suggested_draft_num = 8;
      generator_->GenerateDraft(input_tokens, step++, suggested_draft_num, draft_tokens, 0, accepted_token, req);
      total_tokens += draft_tokens.size();
      accepted_token = 0;
      for (size_t i = 0; i < draft_tokens.size(); i++) {
        if (pos < response.size() && draft_tokens[i] == response[pos]) {
          input_tokens.push_back(response[pos]);
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
        input_tokens.push_back(response[pos]);
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

// Tests for LlmRuntime::DraftTokenFilter
// Mock StructuredGeneratorInterface for testing
class MockStructuredGenerator : public StructuredGeneratorInterface {
 public:
  explicit MockStructuredGenerator(const std::vector<int>& accepted_tokens) : accepted_tokens_(accepted_tokens) {}

  bool AcceptToken(int token_id) override {
    // 只接受在 accepted_tokens_ 列表中的 token
    return std::find(accepted_tokens_.begin(), accepted_tokens_.end(), token_id) != accepted_tokens_.end();
  }

  bool FillNextTokenBitmask(void* next_token_bitmask) override { return true; }

  bool FindJumpForwardTokens(int& rollback_token_num, std::vector<int>& jump_tokens) override { return false; }

  bool IsTerminated() const override { return false; }

  bool IsValid() const override { return true; }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::NONE; }

 private:
  std::vector<int> accepted_tokens_;
};

class DraftTokenFilterTest : public testing::Test {
 protected:
  void SetUp() override {
    runtime_config_ = RuntimeConfig();
    batch_scheduler_config_ = BatchSchedulerConfig();
    context_ = std::make_shared<Context>(1, 1, 1);  // tp_size=1, attn_dp_size=1, max_multi_batch=1
    llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  }

  std::shared_ptr<InferRequest> CreateMockInferRequest(
      const std::vector<int>& draft_tokens_mtp, const std::vector<int>& draft_tokens_trie,
      const std::vector<int>& sampling_result_tokens, const std::vector<int>& stop_token_ids = {},
      std::shared_ptr<StructuredGeneratorInterface> structured_generator = nullptr) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    python_input->model_name = "test_model";
    python_input->input_tokens = {1, 2, 3};
    python_input->sampling_config.stop_token_ids = stop_token_ids;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(python_input, req_ctx);

    request->req_id = 1;
    request->model_name = "test_model";
    request->input_tokens = {1, 2, 3};
    request->output_tokens = {};
    request->logprobs = {};
    request->sampling_config.stop_token_ids = stop_token_ids;
    request->finished = false;
    request->aborted = false;

    auto infer_req = std::make_shared<InferRequest>(request, 0);
    infer_req->draft_tokens.mtp = draft_tokens_mtp;
    infer_req->draft_tokens.trie = draft_tokens_trie;
    infer_req->sampling_result_tokens = sampling_result_tokens;
    infer_req->structured_generator = structured_generator;

    return infer_req;
  }

  RuntimeConfig runtime_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<LlmRuntime> llm_runtime_;
};

TEST_F(DraftTokenFilterTest, StructuredGeneratorRejectsNewToken) {
  // 测试 structured_generator 拒绝新生成 token 的情况
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto structured_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{20});
  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, {}, structured_generator);
  std::vector<std::shared_ptr<InferRequest>> reqs = {req};

  llm_runtime_->DraftTokenFilter(reqs);

  EXPECT_EQ(req->accepted_tokens.size(), 1);  // draft_hit_num
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->generated_token, 20);
}

TEST_F(DraftTokenFilterTest, StructuredGeneratorAcceptsAllTokens) {
  // 测试 structured_generator 接受所有 tokens 的情况
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto structured_generator = std::make_shared<MockStructuredGenerator>(std::vector<int>{20, 30, 40});
  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, {}, structured_generator);
  std::vector<std::shared_ptr<InferRequest>> reqs = {req};

  llm_runtime_->DraftTokenFilter(reqs);

  EXPECT_EQ(req->accepted_tokens.size(), 3);
  EXPECT_EQ(req->accepted_tokens[1], 20);
  EXPECT_EQ(req->accepted_tokens[2], 30);
  EXPECT_EQ(req->generated_token, 40);
}
}  // namespace ksana_llm
