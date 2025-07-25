/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler_test.h"

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include <memory>
#include <utility>
#include <vector>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// 创建一个派生类，方便编辑测试
class TestBatchScheduler : public BatchScheduler {
 public:
  TestBatchScheduler(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                     std::vector<std::shared_ptr<ModelInstance>> req_group)
      : BatchScheduler(batch_scheduler_config, runtime_config, req_group) {}

  void AddToWaitingReqs(std::vector<std::shared_ptr<InferRequest>>& requests) {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    for (auto& req : requests) {
      waiting_reqs_.push_back(req);
    }
  }

  void ClearDpWaitingReqs() {
    for (auto& reqs : dp_waiting_reqs_) {
      reqs.clear();
    }
  }
};

class BalanceReqsTest : public BatchSchedulerTest {
 protected:
  void CommonSetUp(int dp_num) {
    BatchSchedulerTest::CommonSetUp(dp_num);
    std::vector<std::shared_ptr<ModelInstance>> model_instance;
    test_batch_scheduler_ = new TestBatchScheduler(batch_scheduler_config_, runtime_config_, model_instance);
  }

  void TearDown() override {
    delete test_batch_scheduler_;
    BatchSchedulerTest::TearDown();
  }

  size_t InitReqs(int reqs_num, std::vector<std::shared_ptr<InferRequest>>& requests, int token_num = 0) {
    size_t total_tokens = 0;
    for (int i = 0; i < reqs_num; i++) {
      std::shared_ptr<Request> req;
      auto infer_req_group = env_simulator_->InitRequest(i, 10, 5, req, {{0, i}});
      auto r = infer_req_group[0];
      int tokens = (token_num > 0) ? token_num : (i + 1);
      r->kv_cached_token_num = i;
      r->forwarding_tokens.resize(i + tokens);
      requests.push_back(r);
      total_tokens += i + 1;
    }
    return total_tokens;
  }

  void ReqsToPairs(std::vector<std::shared_ptr<InferRequest>>& requests,
                   std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>>& pairs) {
    pairs.clear();
    pairs.reserve(requests.size());
    for (auto req : requests) {
      int token_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      pairs.push_back(std::make_pair(token_num, req));
    }
  }

 protected:
  TestBatchScheduler* test_batch_scheduler_ = nullptr;
};

TEST_F(BalanceReqsTest, BasicAlgoTest) {
  int dp_num = 3;
  CommonSetUp(dp_num);
  BalanceReqsAlgo algo;

  // 创建测试数据
  std::vector<float> workloads = {5.0, 6.0, 7.0};  // 三个处理组的初始负载

  // 创建一些请求
  std::vector<std::shared_ptr<InferRequest>> requests;
  int reqs_num = dp_num * 5 + 1;
  size_t total_tokens = InitReqs(reqs_num, requests);
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> total_reqs;
  ReqsToPairs(requests, total_reqs);

  std::vector<std::vector<std::shared_ptr<InferRequest>>> dp_waiting_reqs;
  algo.BalanceReqs(workloads, total_reqs, dp_waiting_reqs);

  // 验证结果
  // 1. 所有请求都应该被分配
  size_t assigned_reqs = 0;
  for (const auto& group : dp_waiting_reqs) {
    assigned_reqs += group.size();
  }
  EXPECT_EQ(assigned_reqs, total_reqs.size());

  // 2. 计算每个组的token总数
  std::vector<size_t> token_sums(dp_num, 0);
  for (int i = 0; i < dp_num; i++) {
    token_sums[i] = workloads[i] * total_tokens / reqs_num;
    for (const auto& req : dp_waiting_reqs[i]) {
      int64_t tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      token_sums[i] += tokens_num;
    }
  }
  // 3. 验证每个组的token总数应该接近相等
  for (int i = 1; i < dp_num; i++) {
    EXPECT_NEAR(token_sums[i], token_sums[0], token_sums[0] * 0.2);
  }
}

TEST_F(BalanceReqsTest, EmptyRequestsTest) {
  int dp_num = 3;
  CommonSetUp(dp_num);
  BalanceReqsAlgo algo;

  std::vector<float> workloads = {1.0, 2.0, 3.0};
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> total_reqs;

  // 确保每个组都有一些初始请求
  std::vector<std::vector<std::shared_ptr<InferRequest>>> dp_waiting_reqs;
  for (auto& group : dp_waiting_reqs) {
    std::shared_ptr<Request> req;
    auto infer_req_group = env_simulator_->InitRequest(100, 10, 5, req, {{0, 100}});
    group.push_back(infer_req_group[0]);
  }

  algo.BalanceReqs(workloads, total_reqs, dp_waiting_reqs);

  // 验证结果：所有组应该被清空
  for (const auto& group : dp_waiting_reqs) {
    EXPECT_EQ(group.size(), 0);
  }
}

TEST_F(BalanceReqsTest, DifferentWorkloadsTest) {
  int dp_num = 3;
  CommonSetUp(dp_num);
  BalanceReqsAlgo algo;

  // 创建测试数据
  std::vector<float> workloads = {1.0, 5.0, 10.0};  // 初始工作负载差异很大

  std::vector<std::shared_ptr<InferRequest>> requests;
  int token_num = 10;
  InitReqs(20, requests, token_num);
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> total_reqs;
  ReqsToPairs(requests, total_reqs);

  std::vector<std::vector<std::shared_ptr<InferRequest>>> dp_waiting_reqs;
  algo.BalanceReqs(workloads, total_reqs, dp_waiting_reqs);

  // 验证结果
  // 1. 所有请求都应该被分配
  size_t total_assigned = 0;
  for (const auto& group : dp_waiting_reqs) {
    total_assigned += group.size();
  }
  EXPECT_EQ(total_assigned, total_reqs.size());

  // 2. 工作负载较低的组应该分配更多的请求
  EXPECT_GE(dp_waiting_reqs[0].size(), dp_waiting_reqs[1].size());
  EXPECT_GE(dp_waiting_reqs[1].size(), dp_waiting_reqs[2].size());
}

TEST_F(BalanceReqsTest, BalanceWaitingReqsTest) {
  int dp_num = 3;
  CommonSetUp(dp_num);
  size_t multi_batch_id = 0;
  // 创建一些请求
  std::vector<std::shared_ptr<InferRequest>> requests;
  InitReqs(dp_num * 5 + 1, requests);
  for (size_t i = 0; i < test_batch_scheduler_->batch_states_.size(); ++i) {
    test_batch_scheduler_->batch_states_[i][multi_batch_id]->waiting_queue.clear();
    test_batch_scheduler_->batch_states_[i][multi_batch_id]->schedule_output->running_reqs.resize(i + 1);
    test_batch_scheduler_->batch_states_[i][multi_batch_id]->swapped_queue.clear();
  }

  // 添加请求到waiting_reqs_
  test_batch_scheduler_->AddToWaitingReqs(requests);
  test_batch_scheduler_->ClearDpWaitingReqs();

  test_batch_scheduler_->BalanceWaitingReqs();

  // 验证结果
  // 1. 所有请求都应该被分配
  size_t total_assigned = 0;
  for (int i = 0; i < dp_num; i++) {
    total_assigned += test_batch_scheduler_->dp_waiting_reqs_[i].size();
  }
  EXPECT_EQ(total_assigned, requests.size());

  // 2. 工作负载较低的组应该分配更多的请求
  EXPECT_GE(test_batch_scheduler_->dp_waiting_reqs_[0].size(), test_batch_scheduler_->dp_waiting_reqs_[1].size());
  EXPECT_GE(test_batch_scheduler_->dp_waiting_reqs_[1].size(), test_batch_scheduler_->dp_waiting_reqs_[2].size());

  EXPECT_TRUE(test_batch_scheduler_->IsIdle(multi_batch_id));
}

}  // namespace ksana_llm
