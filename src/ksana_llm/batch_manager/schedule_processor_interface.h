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
  std::shared_ptr<std::map<ModelInstance *, std::vector<ForwardRequest *>>> grouped_reqs;
  std::shared_ptr<std::unordered_map<int64_t, std::shared_ptr<std::vector<int>>>> deep_copy_forwarding_tokens;
  std::shared_ptr<std::vector<SamplingRequest>> sampling_reqs;
  bool is_valid = false;
};

class ScheduleProcessorInterface {
 public:
  virtual ~ScheduleProcessorInterface() = default;

  // 初始化处理器
  virtual void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler,
                          std::shared_ptr<LlmRuntime> llm_runtime,
                          std::shared_ptr<MultiBatchController> multi_batch_controller) = 0;

  // 获取下一个调度结果（核心接口）
  // 同步：直接调度+处理数据+返回完整结果
  // 异步：从队列获取已完全处理的结果
  // 这个方法返回的结果应该是完全可用的，不需要额外处理
  virtual ScheduleResult GetNextScheduleResult(size_t multi_batch_id) = 0;

  // 启动和停止
  virtual void Start() = 0;
  virtual void Stop() = 0;

 protected:
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_;
  std::shared_ptr<LlmRuntime> llm_runtime_;
  std::shared_ptr<MultiBatchController> multi_batch_controller_;
};

}  // namespace ksana_llm