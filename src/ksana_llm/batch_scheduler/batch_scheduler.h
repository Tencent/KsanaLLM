/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "ksana_llm/batch_scheduler/batch_scheduler_balance_reqs_algo.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_interface.h"
#include "ksana_llm/batch_scheduler/state/batch_state.h"
#include "ksana_llm/batch_scheduler/strategy/strategy_factory.h"
#include "ksana_llm/batch_scheduler/workload_balance/pp_multibatch_balancer.h"

#include "ksana_llm/cache_manager/cache_manager_interface.h"

#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class BatchScheduler : public BatchSchedulerInterface {
 public:
  BatchScheduler(const BatchSchedulerConfig &batch_scheduler_config, int dp_num, int tp_num);
  ~BatchScheduler();

  // Get the next infer reqs that ready to run.
  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t pp_batch_idx) override;

  // Add infer request to waiting list.
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group) override;

  // Set the cache manager instance of batch scheduler.
  void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int attn_dp_idx) override;

  std::shared_ptr<CacheManagerInterface> &GetCacheManager(int attn_dp_idx) override;

  // Whether the scheduler is idle, that is, waiting buffer and swapped queue is both empty.
  bool IsIdle(size_t pp_batch_idx) override;

  void WaitUntilHaveReqs(size_t pp_batch_idx) override;

  void Stop() override;

 private:
  // Add infer requests to waiting buffer queue, and reject requests if the queue is full.
  Status EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>> &infer_request_group);

  // True if request length exceed the max input length.
  inline bool CheckRequestExceedLength(const std::shared_ptr<InferRequest> req);

  // balance waiting reqs to dp_waiting_reqs_ by batch_state_
  // the output is dp_waiting_reqs_
  void BalanceWaitingReqs();

  void BalancePPMultiBatchReqs(size_t pp_batch_idx);

  void ReportBatchState(std::shared_ptr<BatchState> batch_state);

 private:
  // The config of batch scheduler.
  BatchSchedulerConfig batch_scheduler_config_;

  size_t dp_num_;
  size_t pp_batch_num_;

  // The thread pool of batch scheduler.
  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

  // The batch state informations, include some queues and mutexes. [dp_idx, pp_batch_idx]
  std::vector<std::vector<std::shared_ptr<BatchState>>> batch_states_;

  // The buffer queue needed be scheduled in strategy.
  std::vector<std::shared_ptr<InferRequest>> waiting_reqs_;
  // Protect the waiting_reqs_.
  std::mutex waiting_reqs_mutex_;

  // The buffer queue needed be scheduled in strategy
  std::vector<std::vector<std::shared_ptr<InferRequest>>> dp_waiting_reqs_;

  // The batch strategy implementations.
  std::vector<std::shared_ptr<BaseScheduleStrategy>> schedule_strategies_;

  // Balance requests algorithm
  std::unique_ptr<BalanceReqsAlgo> balance_reqs_algo_ = nullptr;

  // Balance requests among multiple batchs in pipeline parallel
  std::unique_ptr<PPMultibatchWorkloadBalancer> pp_multibatch_wl_balancer_ = nullptr;

  // NOTE(karlluo, jackyjtang): The thread pool is not thread safe, so we need to be temp variable
  // group of all strategy schedule outputs
  std::shared_ptr<ScheduleOutputGroup> schedule_output_group_;

  std::mutex schedule_mutex_;

  bool terminating_ = false;
};

}  // namespace ksana_llm
