/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/batch_scheduler/batch_scheduler_interface.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/sampling_request.h"

namespace ksana_llm {

// 调度结果结构体
struct ScheduleResult {
  std::shared_ptr<ScheduleOutput> schedule_output;
  std::map<ModelInstance *, std::vector<ForwardRequest *>> grouped_reqs;
  std::unordered_map<int64_t, std::shared_ptr<std::vector<int>>> deep_copy_forwarding_tokens;
  std::vector<SamplingRequest*> sampling_reqs;
  GenerationOutputGroup generation_output_group;

  std::vector<ScheduleOutput *> outputs;  // keep ref for async info clearing
};

class ScheduleProcessorInterface {
 public:
  virtual ~ScheduleProcessorInterface() = default;

  // 初始化处理器
  virtual void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler,
                          std::shared_ptr<LlmRuntime> llm_runtime,
                          std::shared_ptr<MultiBatchController> multi_batch_controller) = 0;

  virtual std::shared_ptr<ScheduleResult> GetNextScheduleResult(size_t multi_batch_id) = 0;

  virtual void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup &generation_output) = 0;

  virtual void Stop() = 0;

 protected:
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_;
  std::shared_ptr<LlmRuntime> llm_runtime_;
  std::shared_ptr<MultiBatchController> multi_batch_controller_;
};

}  // namespace ksana_llm
