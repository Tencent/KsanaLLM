/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_manager/schedule_processor_interface.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The schedule processor.
class ScheduleProcessor : public ScheduleProcessorInterface {
 public:
  ScheduleProcessor() = default;
  ~ScheduleProcessor() override = default;

  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler, std::shared_ptr<LlmRuntime> llm_runtime,
                  std::shared_ptr<MultiBatchController> multi_batch_controller) override;

  // The sync mode: call Schedule directly -> check for running requests -> wait or process data
  ScheduleResult GetNextScheduleResult(size_t multi_batch_id) override;

  void Start() override;
  void Stop() override;

 private:
  // Internal helper method: processes schedule data.
  Status ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result);

 private:
  std::atomic<bool> terminated_{false};
};

}  // namespace ksana_llm
