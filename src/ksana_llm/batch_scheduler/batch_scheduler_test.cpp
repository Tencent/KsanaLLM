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
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 1, max_input_num, req_list);
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
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 1, max_input_num, req_list);
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
  int request_num = 10;
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
  EXPECT_EQ(stat.swapout_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SplitFusePressTest) {
  KLLM_LOG_INFO << "SplitFusePressTest start";
  split_fuse_token_num_ = 16;
  CommonSetUp();
  ParallelTester tester(batch_scheduler_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  ParallelTester::SplitFuseCheckHook split_fuse_check_hook(split_fuse_token_num_);
  hooks.push_back(&default_hook);
  hooks.push_back(&split_fuse_check_hook);

  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  // exceed cache size, not exceed_batch size and max_step_size
  int request_num = 10;
  int client_num = 10;
  int min_expect_output_num = 1;
  int max_expect_output_num = 40;
  int min_input_num = 17;
  int max_input_num = 60;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
                          req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, FixPrefixCacheNoSwapTriggeredTest) {
  enable_prefix_cache_ = true;
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, false);
  test_case.SetBatchScheduler(batch_scheduler_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(BatchSchedulerTest, FixPrefixCacheNoSwapTriggeredSplitfuseTest) {
  enable_prefix_cache_ = true;
  split_fuse_token_num_ = 6;
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, false);
  test_case.SetBatchScheduler(batch_scheduler_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(BatchSchedulerTest, FixPrefixCacheSwapTriggeredTest) {
  enable_prefix_cache_ = true;
  CommonSetUp();

  int prefix_block_num = 30;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, false);
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
  req_list[0].req->timestamp_in_us = 1;
  req_list[0].infer_req_group[0]->req_id = 128;
  req_list[0].infer_req_group[0]->timestamp_in_us = 1;
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

  auto mock_requests = batch_scheduler->GetMockRequest();
  auto mock_req = mock_requests[0];

  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  const size_t mock_request_length = runtime_config.mtp_step_num + 1 + 1;

  // Verify initial state after MockRequest creation
  EXPECT_EQ(mock_req->kv_cache_blocks.size(), tp_num);
  EXPECT_EQ(mock_req->infer_stage, InferStage::kContext);
  EXPECT_EQ(mock_req->step, 0);
  EXPECT_EQ(mock_req->kv_cached_token_num, 0);
  EXPECT_FALSE(mock_req->finished);
  EXPECT_TRUE(mock_req->finish_status.OK());

  // Verify tokens size is reasonable
  EXPECT_LE(mock_req->output_tokens.size(), mock_request_length + 10);
}

// Test MockRequest continuous scheduling cycle
TEST_F(BatchSchedulerTest, MockRequestContinuousSchedulingTest) {
  KLLM_LOG_INFO << "BatchSchedulerTest: MockRequestContinuousSchedulingTest";

  int dp_num = 1;
  int tp_num = 1;
  int ep_world_size = 2;
  CommonSetUp(dp_num, tp_num, ep_world_size);

  BatchScheduler* batch_scheduler = static_cast<BatchScheduler*>(batch_scheduler_);

  // First schedule - no running requests
  auto schedule_output1 = batch_scheduler->Schedule(0);
  EXPECT_EQ(schedule_output1->RunningSize(), 0);

  // Second schedule - MockRequest should be scheduled
  auto schedule_output2 = batch_scheduler->Schedule(0);
  EXPECT_EQ(schedule_output2->RunningSize(), 1);

  // Get the scheduled mock request
  ASSERT_GT(schedule_output2->outputs.size(), 0);
  auto& running_reqs = schedule_output2->outputs[0]->running_reqs;
  ASSERT_EQ(running_reqs.size(), 1);
  auto scheduled_mock_req = running_reqs[0];
  EXPECT_TRUE(scheduled_mock_req->is_mock_req);

  // Simulate inference completion by marking as finished
  scheduled_mock_req->finished = true;
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  const size_t mock_request_length = runtime_config.mtp_step_num + 1 + 1;

  for (size_t i = 0; i < mock_request_length; i++) {
    scheduled_mock_req->output_tokens.push_back(i);
  }

  // Update with generation result
  GenerationOutputGroup gen_output;
  gen_output.BuildFromScheduleOutputGroup(*schedule_output2);
  batch_scheduler->UpdateWithGenerationResult(0, gen_output);

  // Third schedule - MockRequest should be rescheduled after recompute
  auto schedule_output3 = batch_scheduler->Schedule(0);

  // Verify MockRequest can be scheduled again (not stuck due to finished flag)
  // This verifies the fix in continuous_batching.cpp:107-109
  int schedule_attempts = 0;
  const int max_attempts = 10;
  bool mock_req_rescheduled = false;

  while (schedule_attempts < max_attempts) {
    auto output = batch_scheduler->Schedule(0);
    if (output->RunningSize() > 0) {
      for (const auto& sched_out : output->outputs) {
        if (sched_out != nullptr) {
          for (const auto& req : sched_out->running_reqs) {
            if (req->is_mock_req) {
              mock_req_rescheduled = true;
              break;
            }
          }
        }
      }
      if (mock_req_rescheduled) break;
    }
    schedule_attempts++;
  }

  EXPECT_TRUE(mock_req_rescheduled)
      << "MockRequest should be reschedulable after recompute, verifying finished flag reset";
}