/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler_test.h"

#include <exception>
#include <memory>
#include <thread>

#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/grammar_backend.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

using namespace ksana_llm;

TEST_F(BatchSchedulerTest, BasicTokenGenerationTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests one by one
  int request_num = 100;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInNotTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // input and max output token are limited, SwapOut/In are not triggered.
  int request_num = 100;
  int client_num = 10;
  int max_expect_output_num = 40;
  int max_input_num = 60;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  // exceed cache size, not exceed_batch size and max_step_size
  int request_num = 100;
  int client_num = 10;
  int min_expect_output_num = 1;
  int max_expect_output_num = 200;
  int min_input_num = 150;
  int max_input_num = 200;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
                          req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_GT(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_GT(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, FixPrefixCacheNoSwapTriggeredTest) {
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, false);
  test_case.SetBatchScheduler(batch_scheduler_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(BatchSchedulerTest, FixPrefixCacheSwapTriggeredTest) {
  CommonSetUp();

  int prefix_block_num = 30;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, false);
  test_case.SetBatchScheduler(batch_scheduler_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestSwapTriggered();
}

TEST_F(BatchSchedulerTest, FixPrefixCacheSwapTriggeredRecomputeTest) {
  CommonSetUp();
  batch_scheduler_config_.max_batch_size = 4;
  batch_scheduler_config_.max_step_token_num = 5;
  block_manager_config_.device_allocator_config.blocks_num = 50;

  int prefix_block_num = 30;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, false);
  test_case.SetBatchScheduler(batch_scheduler_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestSwapTriggered();
}

TEST_F(BatchSchedulerTest, CheckRequestTimeoutTest) {
  CommonSetUp();
  batch_scheduler_config_.waiting_timeout_in_ms = 0;
  ParallelTester tester(batch_scheduler_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests one by one
  int request_num = 1;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);

  req_list[0].req_id = 128;
  req_list[0].req->req_id = 128;
  req_list[0].req->timestamp_in_ms = 1;
  req_list[0].infer_req_group[0]->req_id = 128;
  req_list[0].infer_req_group[0]->timestamp_in_ms = 1;
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  EXPECT_EQ(req_list[0].req->finish_status.GetCode(), RET_REQUEST_TIMEOUT);
}

TEST_F(BatchSchedulerTest, CreateMockRequest) {
  KLLM_LOG_INFO << "BatchSchedulerTest: CreateMockRequest";

  int dp_num = 1;
  int tp_num = 1;
  int ep_world_size = 2;
  CommonSetUp(dp_num, tp_num, ep_world_size);
  BatchScheduler* batch_scheduler = static_cast<BatchScheduler*>(batch_scheduler_);
  EXPECT_EQ(batch_scheduler->GetMockRequest().size(), 1);

  std::shared_ptr<ScheduleOutputGroup> schedule_output_group = batch_scheduler->Schedule(0);
  EXPECT_EQ(schedule_output_group->RunningSize(), 0);

  schedule_output_group = batch_scheduler->Schedule(0);
  EXPECT_EQ(schedule_output_group->RunningSize(), 1);
}


TEST_F(BatchSchedulerTest, NotifyAsyncFinishedRequestsBasicTest) {
  KLLM_LOG_INFO << "BatchSchedulerTest: NotifyAsyncFinishedRequestsBasicTest";

  CommonSetUp();
  BatchScheduler* batch_scheduler = static_cast<BatchScheduler*>(batch_scheduler_);

  // 测试基本的 NotifyAsyncFinishedRequests 调用
  // 在没有异步完成请求的情况下，应该能正常执行而不出错
  EXPECT_NO_THROW(batch_scheduler->NotifyAsyncFinishedRequests());
}
