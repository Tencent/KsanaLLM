/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <memory>
#include <utility>
#include <vector>

#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/batch_scheduler/workload_balance/pp_multibatch_balancer.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {
class PPMultibatchWorkloadBalancerTest : public testing::Test {
 protected:
  void SetUp() override {
    setenv("KLLM_LOG_LEVEL", "SCHEDULER", 1);
    InitLoguru();
    InitializeScheduleOutputPool();
  }

  void TearDown() override { DestroyScheduleOutputPool(); }
};

TEST_F(PPMultibatchWorkloadBalancerTest, ReqWbTest) {
  size_t pp_max_batch_num = 2;
  size_t multi_batch_id = 1;
  BatchSchedulerConfig batch_scheduler_config;
  batch_scheduler_config.max_batch_size = 128;
  batch_scheduler_config.max_waiting_queue_len = 32;

  std::vector<std::shared_ptr<BatchState>> batch_states;
  for (size_t i = 0; i < pp_max_batch_num; i++) {
    batch_states.push_back(std::make_shared<BatchState>(i, batch_scheduler_config));
  }

  // waiting_reqs.size=1, batch 0 has 1 decoding reqs, batch 1 has 9 decoding reqs and 2 waiting reqs
  // expect:
  //     (1) 1 decoding reqs and all waiting reqs in batch 1 are moved to batch 0
  //     (2) decoding reqs in batch 0 are not changed
  //     (3) waiting_reqs.size=0, the request is in waiting reqs of batch 0
  int input_token_num = 10;                           // not used, give any valid value
  int expected_output_token_num = 10;                 // not used, give any valid value
  std::vector<std::pair<int, int>> seeds = {{0, 0}};  // not used, give any valid value
  int tp_num = 1;
  constexpr size_t kBlockNum = 16;

  std::shared_ptr<Request> batch0_req0;
  size_t batch1_req_num = 11;
  size_t batch1_decoding_req_num = 9;
  std::vector<std::shared_ptr<Request>> batch1_reqs{batch1_req_num};
  std::vector<std::shared_ptr<InferRequest>> infer_req_list, waiting_reqs;
  std::shared_ptr<Request> waiting_req;
  int req_id = 0;
  infer_req_list =
      InitFakeRequest(req_id++, input_token_num, expected_output_token_num, batch0_req0, seeds, tp_num, kBlockNum);
  batch_states[0]->decoding_queue.push_back(infer_req_list[0]);
  for (size_t i = 0; i < batch1_req_num; i++) {
    infer_req_list =
        InitFakeRequest(req_id++, input_token_num, expected_output_token_num, batch1_reqs[i], seeds, tp_num, kBlockNum);
    if (i < batch1_decoding_req_num) {
      batch_states[1]->decoding_queue.push_back(infer_req_list[0]);
    } else {
      batch_states[1]->waiting_queue.push_back(infer_req_list[0]);
    }
  }
  infer_req_list =
      InitFakeRequest(req_id++, input_token_num, expected_output_token_num, waiting_req, seeds, tp_num, kBlockNum);
  waiting_reqs.push_back(infer_req_list[0]);

  PPMultibatchWorkloadBalancer balancer(PPMultibatchWBStrategy::WB_BATCH_REQ);
  balancer.BalancePPMultiBatchReqs(multi_batch_id, waiting_reqs, batch_states);

  // batch0 decoding reqs should not be changed
  EXPECT_EQ(batch_states[0]->decoding_queue.size(), 3);
  EXPECT_EQ(batch_states[0]->decoding_queue[0]->req_id, 0);

  // total 9 reqs. 2 decoding reqs are moved to batch 0, waiting_reqs are moved to batch 0
  EXPECT_EQ(batch_states[0]->waiting_queue.size(), 3);
  EXPECT_EQ(batch_states[1]->waiting_queue.size(), 0);
  EXPECT_EQ(batch_states[1]->decoding_queue.size(), batch1_decoding_req_num - 2);
}
}  // namespace ksana_llm
