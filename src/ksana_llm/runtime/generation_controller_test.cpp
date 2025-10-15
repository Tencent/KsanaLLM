/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/generation_controller.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace ksana_llm {

// Mock StructuredGeneratorInterface for testing
class MockStructuredGenerator : public StructuredGeneratorInterface {
 public:
  explicit MockStructuredGenerator(const std::vector<int>& expected_output_tokens)
      : expected_output_tokens_(expected_output_tokens) {}

  bool AcceptToken(int token_id) override {
    if (expecting_idx_ >= expected_output_tokens_.size()) {
      return false;
    }
    // 只能顺序接收 expected_output_tokens_ 列表中的 token
    if (expected_output_tokens_[expecting_idx_] == token_id) {
      KLLM_LOG_DEBUG << "Accept token: " << token_id << ", expecting idx: " << expecting_idx_;
      expecting_idx_++;
      return true;
    }
    return false;
  }

  bool FillNextTokenBitmask(void* next_token_bitmask) override { return true; }

  void Rollback(int rollback_token_num) override {
    KLLM_CHECK(rollback_token_num < expecting_idx_);
    expecting_idx_ -= rollback_token_num;
  }

  bool FindJumpForwardTokens(std::vector<int>& jump_tokens) override { return false; }

  bool IsTerminated() const override { return false; }

  bool IsValid() const override { return true; }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::REGEX; }

 private:
  std::vector<int> expected_output_tokens_;
  size_t expecting_idx_ = 0;
};

class MockGeneratorCreator : public GeneratorCreator {
 public:
  explicit MockGeneratorCreator(const std::vector<int>& expected_output_tokens)
      : expected_output_tokens_(expected_output_tokens) {}
  std::shared_ptr<StructuredGeneratorInterface> CreateGenerator(const StructuredGeneratorConfig& config) override {
    return std::make_shared<MockStructuredGenerator>(expected_output_tokens_);
  }

  StructuredConstraintType GetConstraintType() const override { return StructuredConstraintType::REGEX; }

 private:
  std::vector<int> expected_output_tokens_;
};

class GenerationControllerTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    std::shared_ptr<StructuredGeneratorFactory> factory = std::make_shared<StructuredGeneratorFactory>();
    std::vector<int> expected_output_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    factory->RegisterCreator(StructuredConstraintType::REGEX,
                             std::make_unique<MockGeneratorCreator>(expected_output_tokens));
    generation_controller_ = std::make_shared<GenerationController>(factory);
  }

  std::shared_ptr<Request> CreateMockRequest(const std::vector<int>& stop_token_ids = {}) {
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

    return request;
  }

  std::shared_ptr<GenerationController> generation_controller_;
};

TEST_F(GenerationControllerTest, InitGenerationState) {
  std::vector<int> draft_mtp;
  std::vector<int> draft_trie;
  std::vector<int> sampling_result;

  auto req = CreateMockRequest();
  std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, 0);
  infer_req->structured_generator_config.constraint_type = StructuredConstraintType::REGEX;
  infer_req->structured_generator_config.constraint_spec = "dummy";
  std::vector<std::shared_ptr<InferRequest>> reqs = {infer_req};

  generation_controller_->InitGenerationState(reqs);
  EXPECT_TRUE(infer_req->structured_generator != nullptr);
  EXPECT_TRUE(infer_req->structured_generator->IsValid());
}

TEST_F(GenerationControllerTest, UpdateGenerationState) {
  // 测试 structured_generator 拒绝新生成 token 的情况

  std::vector<int> expected_output_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto structured_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);
  auto req = CreateMockRequest();
  std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, 0);
  infer_req->structured_generator = structured_generator;
  std::vector<std::shared_ptr<InferRequest>> reqs = {infer_req};

  // First sampling_result_tokens is acceptable, generated_token is sampling_result_tokens[0];
  infer_req->sampling_result_tokens = {1};
  infer_req->draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->generated_token, 1);

  // Expecting 2 to be generated, but sampling result is 1, so generated_token is -1;
  infer_req->sampling_result_tokens = {1};
  infer_req->draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  //  EXPECT_EQ(infer_req->generated_token, -1);  // TODO(robertyuan): scheduler doesn't handle this value. handle this
  //  after generated_token is changed to a vector.

  // Expecting 2,3 to be generated, draft_tokens is 2 and sampling result is 2,3, so accepted_token is 2,
  // generated_token is 3;
  infer_req->sampling_result_tokens = {2, 3};
  infer_req->draft_tokens.mtp = {2};
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 1);
  EXPECT_EQ(infer_req->accepted_tokens[0], 2);
  EXPECT_EQ(infer_req->generated_token, 3);

  // Expecting 4,5,6 to be generated, draft_tokens is 4,5 and sampling result is 4,7,8, so accepted_token is null
  // because generated token 7 is rejected by structure generator, generated_token is 4;
  // computing for draft token 4 is wasted.
  infer_req->sampling_result_tokens = {4, 7, 8};
  infer_req->draft_tokens.mtp = {4, 5};
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->generated_token, 4);

  // Expecting 5,6,7 to be generated, draft_tokens is 5,6 and sampling result is 5,6,7, so accepted_token is 5,6 because
  // 6,7 are accepted by structure generator, generated_token is 7;
  infer_req->sampling_result_tokens = {5, 6, 7};
  infer_req->draft_tokens.mtp = {5, 6};
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 2);
  EXPECT_EQ(infer_req->accepted_tokens[0], 5);
  EXPECT_EQ(infer_req->accepted_tokens[1], 6);
  EXPECT_EQ(infer_req->generated_token, 7);
}
}  // namespace ksana_llm
