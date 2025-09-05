/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace ksana_llm;

class MockTransferEngine : public TransferEngine {
 public:
  MOCK_METHOD(void, Send, ((std::vector<std::tuple<std::string, int, std::vector<int>>>&)), (override));

  static std::shared_ptr<MockTransferEngine> GetInstance() { return Singleton<MockTransferEngine>::GetInstance(); }
  static void DeleteInstance() { Singleton<MockTransferEngine>::DeleteInstance(); }
};

class LlmRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  }

  void TearDown() override { MockTransferEngine::DeleteInstance(); }
  BatchSchedulerConfig batch_scheduler_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_ = std::make_shared<Context>(1, 1, 1);
  std::shared_ptr<LlmRuntime> llm_runtime_ = nullptr;
};

TEST_F(LlmRuntimeTest, TransferGeneratedTokenTest) {
  std::vector<std::shared_ptr<InferRequest>> reqs;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  auto req1 = std::make_shared<InferRequest>(request, 0);
  req1->kv_comm_group_key = "group_key_0";
  req1->kv_comm_request_id = 110;
  req1->generated_token = 66;
  // first req : has draft token
  req1->draft_tokens.mtp.push_back(15);
  reqs.push_back(req1);

  auto req2 = std::make_shared<InferRequest>(request, 0);
  req2->kv_comm_group_key = "group_key_0";
  req2->kv_comm_request_id = 111;
  req2->generated_token = 77;
  // second req : no draft token
  reqs.push_back(req2);

  // expected params
  std::vector<int> req1_send_tokens(MAX_TRANSFER_TOKENS, -1);
  req1_send_tokens[0] = 66;
  req1_send_tokens[1] = 15;
  std::vector<int> req2_send_tokens(MAX_TRANSFER_TOKENS, -1);
  req2_send_tokens[0] = 77;

  std::vector<std::tuple<std::string, int, std::vector<int>>> expected = {
      std::make_tuple("group_key_0", 110, req1_send_tokens),
      std::make_tuple("group_key_0", 111, req2_send_tokens),
  };

  EXPECT_CALL(*MockTransferEngine::GetInstance(), Send(expected)).Times(1);

  llm_runtime_->TransferGeneratedToken(reqs, MockTransferEngine::GetInstance());
}