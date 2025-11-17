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

class LlmRuntimeMock : public LlmRuntime {
  using LlmRuntime::LlmRuntime;

  Status Forward(size_t multi_batch_id, std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                 bool epilogue, RunMode run_mode = RunMode::kMain) override {
    return Status();
  }
  Status Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                  std::vector<SamplingRequest>& sampling_reqs, bool enable_main_layers_sampler = true) override {
    for (auto& req : sampling_reqs) {
      req.sampling_result_tokens->emplace_back(kMockSamplingToken);
    }
    return Status();
  }

 public:
  static constexpr int kMockSamplingToken = 1024;
};

class LlmRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);

    llm_runtime_ = std::make_shared<LlmRuntimeMock>(batch_scheduler_config_, runtime_config_, context_);
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
  req1->generated_tokens = {66};
  // first req : has draft token
  req1->draft_tokens.mtp.push_back(15);
  reqs.push_back(req1);

  auto req2 = std::make_shared<InferRequest>(request, 0);
  req2->kv_comm_group_key = "group_key_0";
  req2->kv_comm_request_id = 111;
  req2->generated_tokens = {77, 88};
  req2->draft_tokens.clear();
  // second req : no draft token
  reqs.push_back(req2);

  // expected params
  std::vector<int> req1_send_tokens(MAX_TRANSFER_TOKENS, -1);
  req1_send_tokens[0] = req1->generated_tokens.size();
  req1_send_tokens[1] = 66;
  req1_send_tokens[2] = req1->draft_tokens.mtp.size();
  req1_send_tokens[3] = 15;
  std::vector<int> req2_send_tokens(MAX_TRANSFER_TOKENS, -1);
  req2_send_tokens[0] = req2->generated_tokens.size();
  req2_send_tokens[1] = 77;
  req2_send_tokens[2] = 88;
  req2_send_tokens[3] = req2->draft_tokens.mtp.size();

  std::vector<std::tuple<std::string, int, std::vector<int>>> expected = {
      std::make_tuple("group_key_0", 110, req1_send_tokens),
      std::make_tuple("group_key_0", 111, req2_send_tokens),
  };

  EXPECT_CALL(*MockTransferEngine::GetInstance(), Send(expected)).Times(1);

  MockTransferEngine::GetInstance()->SetGroupRole(GroupRole::PREFILL);
  llm_runtime_->TransferGeneratedToken(reqs, MockTransferEngine::GetInstance());
}

TEST_F(LlmRuntimeTest, ReorderInferRequestsTest) {
  std::vector<std::shared_ptr<InferRequest>> reqs;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  // 创建多个测试请求，覆盖各种排序场景
  // 排序规则：
  // 1. 按 attn_dp_group_id 升序
  // 2. 在同一组内，prefill 在前，decode 在后
  // 3. 同类型内按 compute_token_num (forwarding_tokens.size() - kv_cached_token_num) 降序
  // 4. compute_token_num 相同时按 kv_cached_token_num 升序
  // 5. 最后按 req_id 升序

  // req1: attn_dp_group_id=0, prefill (kv_cached_token_num=0), compute_token_num=10, req_id=1
  auto req1 = std::make_shared<InferRequest>(request, 0);
  req1->req_id = 1;
  req1->attn_dp_group_id = 0;
  req1->kv_cached_token_num = 0;
  req1->forwarding_tokens = std::vector<int>(10, 1);  // size=10
  reqs.push_back(req1);

  // req2: attn_dp_group_id=0, decode (kv_cached_token_num=10), compute_token_num=1, req_id=2
  auto req2 = std::make_shared<InferRequest>(request, 0);
  req2->req_id = 2;
  req2->attn_dp_group_id = 0;
  req2->kv_cached_token_num = 10;
  req2->forwarding_tokens = std::vector<int>(11, 2);  // size=11, compute_token_num=11-10=1
  reqs.push_back(req2);

  // req3: attn_dp_group_id=1, prefill (kv_cached_token_num=0), compute_token_num=5, req_id=3
  auto req3 = std::make_shared<InferRequest>(request, 0);
  req3->req_id = 3;
  req3->attn_dp_group_id = 1;
  req3->kv_cached_token_num = 0;
  req3->forwarding_tokens = std::vector<int>(5, 3);  // size=5
  reqs.push_back(req3);

  // req4: attn_dp_group_id=0, prefill (kv_cached_token_num=0), compute_token_num=20, req_id=4
  auto req4 = std::make_shared<InferRequest>(request, 0);
  req4->req_id = 4;
  req4->attn_dp_group_id = 0;
  req4->kv_cached_token_num = 0;
  req4->forwarding_tokens = std::vector<int>(20, 4);  // size=20
  reqs.push_back(req4);

  // req5: attn_dp_group_id=0, decode (kv_cached_token_num=20), compute_token_num=1, req_id=5
  auto req5 = std::make_shared<InferRequest>(request, 0);
  req5->req_id = 5;
  req5->attn_dp_group_id = 0;
  req5->kv_cached_token_num = 20;
  req5->forwarding_tokens = std::vector<int>(21, 5);  // size=21, compute_token_num=21-20=1
  reqs.push_back(req5);

  // req6: attn_dp_group_id=0, decode (kv_cached_token_num=20), compute_token_num=1, req_id=6
  auto req6 = std::make_shared<InferRequest>(request, 0);
  req6->req_id = 6;
  req6->attn_dp_group_id = 0;
  req6->kv_cached_token_num = 20;
  req6->forwarding_tokens = std::vector<int>(21, 6);  // size=21, compute_token_num=21-20=1
  reqs.push_back(req6);

  // req7: attn_dp_group_id=0, prefill (kv_cached_token_num=0), compute_token_num=10, req_id=7
  auto req7 = std::make_shared<InferRequest>(request, 0);
  req7->req_id = 7;
  req7->attn_dp_group_id = 0;
  req7->kv_cached_token_num = 0;
  req7->forwarding_tokens = std::vector<int>(10, 7);  // size=10
  reqs.push_back(req7);

  // req8: attn_dp_group_id=1, decode (kv_cached_token_num=10), compute_token_num=1, req_id=8
  auto req8 = std::make_shared<InferRequest>(request, 0);
  req8->req_id = 8;
  req8->attn_dp_group_id = 1;
  req8->kv_cached_token_num = 10;
  req8->forwarding_tokens = std::vector<int>(11, 8);  // size=11, compute_token_num=11-10=1
  reqs.push_back(req8);

  // req9: attn_dp_group_id=0, decode (kv_cached_token_num=5), compute_token_num=1, req_id=9
  auto req9 = std::make_shared<InferRequest>(request, 0);
  req9->req_id = 9;
  req9->attn_dp_group_id = 0;
  req9->kv_cached_token_num = 5;
  req9->forwarding_tokens = std::vector<int>(6, 9);  // size=6, compute_token_num=6-5=1
  reqs.push_back(req9);

  // 调用排序函数
  llm_runtime_->ReorderInferRequests(reqs);

  // 验证排序结果
  // 预期顺序：
  // Group 0 (attn_dp_group_id=0):
  //   Prefill: req4 (compute_token_num=20) -> req1 (compute_token_num=10, req_id=1) -> req7 (compute_token_num=10,
  //   req_id=7) Decode: req9 (kv_cached_token_num=5) -> req2 (kv_cached_token_num=10, req_id=2) -> req5
  //   (kv_cached_token_num=20, req_id=5) -> req6 (kv_cached_token_num=20, req_id=6)
  // Group 1 (attn_dp_group_id=1):
  //   Prefill: req3 (compute_token_num=5)
  //   Decode: req8 (compute_token_num=1)

  std::vector<int64_t> expected_order = {4, 1, 7, 9, 2, 5, 6, 3, 8};

  ASSERT_EQ(reqs.size(), expected_order.size());

  for (size_t i = 0; i < reqs.size(); ++i) {
    EXPECT_EQ(reqs[i]->req_id, expected_order[i])
        << "Position " << i << ": expected req_id " << expected_order[i] << ", but got " << reqs[i]->req_id;
  }
}

TEST_F(LlmRuntimeTest, WorkerBuildForwardRequestsTest) {
  ModelConfig model_config;
  model_config.name = "test_model";
  std::shared_ptr<WeightInstanceInterface> weight_instance = nullptr;
  std::shared_ptr<ModelInstance> model_instance =
      std::make_shared<ModelInstance>(model_config, runtime_config_, context_, weight_instance);

  std::vector<std::shared_ptr<WorkerInferRequest>> reqs;
  auto req1 = std::make_shared<WorkerInferRequest>();
  req1->req_id = 101;
  req1->infer_stage = InferStage::kContext;
  req1->step = 0;
  req1->kv_cached_token_num = 10;
  req1->model_instance = model_instance;
  req1->forwarding_tokens = {1, 2, 3, 4, 5};
  req1->is_use_prefix_cache = true;
  req1->prefix_cache_len = 5;
  req1->attn_dp_group_id = 0;
  reqs.emplace_back(req1);

  auto req2 = std::make_shared<WorkerInferRequest>();
  req2->req_id = 102;
  req2->infer_stage = InferStage::kDecode;
  req2->step = 0;
  req2->kv_cached_token_num = 20;
  req2->model_instance = model_instance;
  req2->forwarding_tokens = {6, 7, 8};
  req2->is_use_prefix_cache = false;
  req2->prefix_cache_len = 0;
  req2->attn_dp_group_id = 1;
  reqs.emplace_back(req2);

  std::map<ModelInstance*, std::vector<ForwardRequest*>> grouped_reqs;
  llm_runtime_->BuildForwardRequests(reqs, grouped_reqs);

  EXPECT_EQ(grouped_reqs.size(), 1);

  std::map<int64_t, ForwardRequest*> results;
  for (auto& [model, forward_reqs] : grouped_reqs) {
    for (auto& forward_req : forward_reqs) {
      results[forward_req->req_id] = forward_req;
    }
  }

  EXPECT_EQ(results.size(), reqs.size());

  const auto& forward_req1 = results[req1->req_id];
  const auto& forward_req2 = results[req2->req_id];

  EXPECT_EQ(forward_req1->step, 1);
  EXPECT_EQ(forward_req2->step, 1);

  EXPECT_EQ(*forward_req1->forwarding_tokens, req1->forwarding_tokens);
  EXPECT_EQ(*forward_req2->forwarding_tokens, req2->forwarding_tokens);
}

TEST_F(LlmRuntimeTest, PrepareMtpInfoTest) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  std::vector<std::shared_ptr<InferRequest>> reqs;
  auto req1 = std::make_shared<InferRequest>(request, 0);
  req1->req_id = 101;
  req1->forwarding_tokens = {1, 2, 3, 4};
  reqs.emplace_back(req1);

  auto req2 = std::make_shared<InferRequest>(request, 0);
  req2->req_id = 102;
  req2->forwarding_tokens = {3, 4, 5, 6};
  reqs.emplace_back(req2);

  WaitGroup wg(1);
  llm_runtime_->PrepareMtpInfo(reqs, wg);
  wg.Wait();
  EXPECT_EQ(llm_runtime_->mtp_prepared_data_.size(), 0);

  *const_cast<size_t*>(&(llm_runtime_->mtp_step_num_)) = 1;
  wg.Add(1);
  llm_runtime_->PrepareMtpInfo(reqs, wg);
  wg.Wait();
  EXPECT_EQ(llm_runtime_->mtp_prepared_data_.size(), 2);

  auto& prepare_1 = llm_runtime_->mtp_prepared_data_[101];
  EXPECT_EQ(prepare_1.order_pos, 0);
  EXPECT_EQ(*prepare_1.tokens, std::vector<int>({2, 3, 4}));

  auto& prepare_2 = llm_runtime_->mtp_prepared_data_[102];
  EXPECT_EQ(prepare_2.order_pos, 1);
  EXPECT_EQ(*prepare_2.tokens, std::vector<int>({4, 5, 6}));
}

TEST_F(LlmRuntimeTest, MtpForwardTest) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  std::vector<std::shared_ptr<InferRequest>> reqs;
  auto req1 = std::make_shared<InferRequest>(request, 0);
  req1->req_id = 101;
  req1->forwarding_tokens = {1, 2, 3, 4};
  req1->generated_tokens = {5};
  reqs.emplace_back(req1);

  auto req2 = std::make_shared<InferRequest>(request, 0);
  req2->req_id = 102;
  req2->forwarding_tokens = {3, 4, 5, 6};
  req2->generated_tokens = {7};
  reqs.emplace_back(req2);

  ForwardRequest forward_req1, forward_req2;
  forward_req1.req_id = req1->req_id;
  forward_req1.kv_cached_token_num = 0;
  forward_req2.req_id = req2->req_id;
  forward_req2.kv_cached_token_num = 0;

  std::vector<ForwardRequest*> forward_reqs;
  forward_reqs.emplace_back(&forward_req1);
  forward_reqs.emplace_back(&forward_req2);

  std::map<ModelInstance*, std::vector<ForwardRequest*>> grouped_reqs;
  grouped_reqs[nullptr] = forward_reqs;

  *const_cast<size_t*>(&(llm_runtime_->mtp_step_num_)) = 2;

  WaitGroup wg(1);
  llm_runtime_->PrepareMtpInfo(reqs, wg);
  wg.Wait();

  std::vector<int> samp_res1, samp_res2;
  std::vector<SamplingRequest> sampling_reqs(reqs.size());
  sampling_reqs[0].sampling_result_tokens = &req1->sampling_result_tokens;
  sampling_reqs[1].sampling_result_tokens = &req2->sampling_result_tokens;

  llm_runtime_->MtpForward(0, grouped_reqs, sampling_reqs, reqs, true);

  constexpr int kSamplingToken = LlmRuntimeMock::kMockSamplingToken;
  EXPECT_EQ(*forward_req1.forwarding_tokens, std::vector<int>({2, 3, 4, 5, kSamplingToken}));
  EXPECT_EQ(forward_req2.kv_cached_token_num, 4);
  EXPECT_EQ(req1->draft_tokens.mtp, std::vector<int>({kSamplingToken, kSamplingToken}));

  EXPECT_EQ(*forward_req2.forwarding_tokens, std::vector<int>({4, 5, 6, 7, kSamplingToken}));
  EXPECT_EQ(forward_req2.kv_cached_token_num, 4);
  EXPECT_EQ(req2->draft_tokens.mtp, std::vector<int>({kSamplingToken, kSamplingToken}));
}
