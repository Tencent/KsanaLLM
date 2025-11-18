/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/generation_controller.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "ksana_llm/runtime/ptp_generation_controller.h"
#include "ksana_llm/runtime/structured_generation/reasoning_structured_generator.h"

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

  // Prefilling, no new token is generated
  infer_req->sampling_result_tokens.clear();
  infer_req->draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->generated_tokens.size(), 0);

  // First sampling_result_tokens is acceptable, generated_token is sampling_result_tokens[0];
  infer_req->sampling_result_tokens = {1};
  infer_req->draft_tokens.clear();
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->generated_tokens.size(), 1);
  EXPECT_EQ(infer_req->generated_tokens[0], 1);

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
  EXPECT_EQ(infer_req->generated_tokens.size(), 1);
  EXPECT_EQ(infer_req->generated_tokens[0], 3);

  // Expecting 4,5,6 to be generated, draft_tokens is 4,5 and sampling result is 4,7,8, so accepted_token is null
  // because generated token 7 is rejected by structure generator, generated_token is 4;
  // computing for draft token 4 is wasted.
  infer_req->sampling_result_tokens = {4, 7, 8};
  infer_req->draft_tokens.mtp = {4, 5};
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->generated_tokens[0], 4);

  // Expecting 5,6,7 to be generated, draft_tokens is 5,6 and sampling result is 5,6,7, so accepted_token is 5,6 because
  // 6,7 are accepted by structure generator, generated_token is 7;
  infer_req->sampling_result_tokens = {5, 6, 7};
  infer_req->draft_tokens.mtp = {5, 6};
  generation_controller_->UpdateGenerationState(reqs);
  EXPECT_EQ(infer_req->accepted_tokens.size(), 2);
  EXPECT_EQ(infer_req->accepted_tokens[0], 5);
  EXPECT_EQ(infer_req->accepted_tokens[1], 6);
  EXPECT_EQ(infer_req->generated_tokens[0], 7);
}

// 测试 ReasoningStructuredGenerator - 有思考内容的模型
// 当 think_end_token_id > 0 时，会包装 ReasoningStructuredGenerator
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorWithReasoning) {
  // 期望的输出序列: [10, 20, 30]
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  // 思考结束标记为 999（> 0，表示支持思考）
  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  // 验证初始状态：处于推理阶段
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->IsTerminated());
  EXPECT_TRUE(reasoner_generator->IsValid());

  // 在推理阶段，任何 token 都应该被接受（模拟思考内容）
  EXPECT_TRUE(reasoner_generator->AcceptToken(1));
  EXPECT_TRUE(reasoner_generator->AcceptToken(2));
  EXPECT_TRUE(reasoner_generator->AcceptToken(3));
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());

  // 遇到思考结束标记，退出推理阶段
  EXPECT_TRUE(reasoner_generator->AcceptToken(think_end_token_id));
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，只接受符合约束的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(10));   // 第一个期望的 token
  EXPECT_FALSE(reasoner_generator->AcceptToken(15));  // 不符合期望的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));   // 第二个期望的 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(30));   // 第三个期望的 token

  // 验证约束类型
  EXPECT_EQ(reasoner_generator->GetConstraintType(), StructuredConstraintType::REGEX);
}

// 测试没有思考内容的模型 - 直接使用Xgrammar 作为约束解码后端
// 这个测试模拟 StructuredGeneratorFactory 在 think_end_token_id <= 0 时的行为
TEST_F(GenerationControllerTest, StructuredGeneratorWithoutReasoning) {
  // 期望的输出序列: [100, 200, 300]
  std::vector<int> expected_output_tokens = {100, 200, 300};

  // 对于不支持思考的模型，直接使用内部生成器，不包装 ReasoningStructuredGenerator
  auto generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  // 验证生成器有效
  EXPECT_TRUE(generator->IsValid());
  EXPECT_TRUE(generator->AcceptToken(100));   // 第一个期望的 token
  EXPECT_FALSE(generator->AcceptToken(150));  // 不符合期望的 token，被拒绝
  EXPECT_TRUE(generator->AcceptToken(200));   // 第二个期望的 token
  EXPECT_FALSE(generator->AcceptToken(250));  // 不符合期望的 token，被拒绝
  EXPECT_TRUE(generator->AcceptToken(300));   // 第三个期望的 token

  // 验证生成器仍然有效
  EXPECT_TRUE(generator->IsValid());
  EXPECT_EQ(generator->GetConstraintType(), StructuredConstraintType::REGEX);
}

// 测试 FillNextTokenBitmask - 推理阶段不应用约束
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorFillNextTokenBitmask) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  void* dummy_bitmask = nullptr;

  // 在推理阶段，FillNextTokenBitmask 应该返回 false（不应用约束）
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  EXPECT_FALSE(reasoner_generator->FillNextTokenBitmask(dummy_bitmask));

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，FillNextTokenBitmask 应该返回 true（应用约束）
  EXPECT_TRUE(reasoner_generator->FillNextTokenBitmask(dummy_bitmask));
}

// 测试 Rollback - 推理阶段不传递回滚
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorRollback) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  // 在推理阶段接受一些 token
  reasoner_generator->AcceptToken(1);
  reasoner_generator->AcceptToken(2);
  reasoner_generator->AcceptToken(3);

  // 在推理阶段回滚不会影响内部生成器（不会抛出异常）
  EXPECT_NO_THROW(reasoner_generator->Rollback(2));
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后接受一些 token
  reasoner_generator->AcceptToken(10);
  reasoner_generator->AcceptToken(20);

  // 推理阶段结束后，回滚会传递给内部生成器
  EXPECT_NO_THROW(reasoner_generator->Rollback(1));

  // 验证回滚后可以重新接受 token
  EXPECT_TRUE(reasoner_generator->AcceptToken(20));

  // 测试负数回滚应该抛出异常
  EXPECT_THROW(reasoner_generator->Rollback(-1), std::invalid_argument);
}

// 测试 FindJumpForwardTokens - 推理阶段不支持跳跃前进
TEST_F(GenerationControllerTest, ReasoningStructuredGeneratorFindJumpForwardTokens) {
  std::vector<int> expected_output_tokens = {10, 20, 30};
  auto inner_generator = std::make_shared<MockStructuredGenerator>(expected_output_tokens);

  int think_end_token_id = 999;
  auto reasoner_generator = std::make_shared<ReasoningStructuredGenerator>(inner_generator, think_end_token_id, true);

  std::vector<int> jump_tokens;

  // 在推理阶段，FindJumpForwardTokens 应该返回 false 并清空 jump_tokens
  EXPECT_TRUE(reasoner_generator->IsInReasoningPhase());
  jump_tokens = {1, 2, 3};
  EXPECT_FALSE(reasoner_generator->FindJumpForwardTokens(jump_tokens));
  EXPECT_TRUE(jump_tokens.empty());

  // 退出推理阶段
  reasoner_generator->AcceptToken(think_end_token_id);
  EXPECT_FALSE(reasoner_generator->IsInReasoningPhase());

  // 推理阶段结束后，FindJumpForwardTokens 会传递给内部生成器
  // MockStructuredGenerator 的实现返回 false
  EXPECT_FALSE(reasoner_generator->FindJumpForwardTokens(jump_tokens));
}

TEST_F(GenerationControllerTest, PTPGenerationControllerTest) {
  std::shared_ptr<PTPGenerationController> ptp_generation_controller =
      std::make_shared<PTPGenerationController>(nullptr);

  auto req = CreateMockRequest();
  std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, 0);
  infer_req->generated_tokens.clear();
  infer_req->sampling_result_tokens = {10, 20, 30};
  infer_req->output_tokens = {1, 2, 3};
  infer_req->forwarding_tokens = {1, 2, 3, 4};

  std::vector<std::shared_ptr<InferRequest>> reqs = {infer_req};

  ptp_generation_controller->UpdateGenerationState(reqs);

  // 采样结果均被认为是可接受，存储于 generated_tokens 中
  EXPECT_EQ(infer_req->generated_tokens.size(), 3);
  EXPECT_EQ(infer_req->generated_tokens[0], 10);
  EXPECT_EQ(infer_req->generated_tokens[1], 20);
  EXPECT_EQ(infer_req->generated_tokens[2], 30);

  // 清空 accepted_tokens 与 draft_tokens
  EXPECT_EQ(infer_req->accepted_tokens.size(), 0);
  EXPECT_EQ(infer_req->draft_tokens.size(), 0);
}

}  // namespace ksana_llm
