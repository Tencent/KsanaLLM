/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/schedule_processor_interface.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The schedule processor.
class ScheduleProcessor : public ScheduleProcessorInterface {
 public:
  explicit ScheduleProcessor(bool enable_async, size_t max_pp_batch_num = 1)
      : enable_async_(enable_async), max_pp_batch_num_(max_pp_batch_num), sched_result_queue_(max_pp_batch_num) {}
  ~ScheduleProcessor() override = default;

  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler, std::shared_ptr<LlmRuntime> llm_runtime,
                  std::shared_ptr<MultiBatchController> multi_batch_controller) override;

  // The sync mode: call Schedule directly -> check for running requests -> wait or process data
  std::shared_ptr<ScheduleResult> GetNextScheduleResult(size_t multi_batch_id) override;

  void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output) override;

  void Stop() override;

 protected:
  void AsyncScheduleThread(size_t multi_batch_id);
  std::shared_ptr<ScheduleResult> Schedule(size_t multi_batch_id);

  void ProcessLaunchableScheduleResult(size_t multi_batch_id, std::shared_ptr<ScheduleResult> result);

  // Internal helper method: processes schedule data.
  Status ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result);

 private:
  std::atomic<bool> terminated_{false};

  bool enable_async_ = false;
  size_t max_pp_batch_num_ = 0;
  // To guard planning results.
  std::mutex planning_result_mutex;
  std::vector<std::unique_ptr<std::thread>> async_sched_threads_;
  std::vector<std::shared_ptr<ScheduleResult>> planning_sched_results_;
  std::vector<BlockingQueue<std::shared_ptr<ScheduleResult>>> sched_result_queue_;
};

}  // namespace ksana_llm
