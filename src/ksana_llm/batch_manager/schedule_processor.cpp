/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/schedule_processor.h"

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/schedule_output_process.h"

namespace ksana_llm {

void ScheduleProcessor::Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler,
                                   std::shared_ptr<LlmRuntime> llm_runtime,
                                   std::shared_ptr<MultiBatchController> multi_batch_controller) {
  batch_scheduler_ = batch_scheduler;
  llm_runtime_ = llm_runtime;
  multi_batch_controller_ = multi_batch_controller;
  KLLM_LOG_INFO << "ScheduleProcessor initialized";
}

// The sync mode: call Schedule directly -> check for running requests -> wait or process data
ScheduleResult ScheduleProcessor::GetNextScheduleResult(size_t multi_batch_id) {
  ScheduleResult result;

  while (!terminated_) {
    // 1. Call Schedule directly
    std::shared_ptr<ScheduleOutputGroup> schedule_output_group = batch_scheduler_->Schedule(multi_batch_id);

    // 2. Merge schedule results
    result.schedule_output = std::make_shared<ScheduleOutput>();
    MergeScheduleOutputGroup(schedule_output_group, *result.schedule_output);

    // 3. Check if there are any running requests
    if (schedule_output_group->RunningSize() == 0) {
      // No running requests, need to wait
      multi_batch_controller_->NotifyCurrentBatchThreadNotReady(multi_batch_id);
      if (batch_scheduler_->IsIdle(multi_batch_id)) {
        batch_scheduler_->WaitUntilHaveReqs(multi_batch_id);
      } else {
        KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " not idle, sleep 100ms";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;  // Continue the loop to wait
    }

    // 4. There are running requests, process the data
    result.is_valid = true;
    if (!ProcessScheduleDataInternal(multi_batch_id, result).OK()) {
      continue;
    }
    return result;
  }
  return result;
}

Status ScheduleProcessor::ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result) {
  if (!result.schedule_output || !llm_runtime_) {
    KLLM_LOG_ERROR << "Invalid schedule_output or llm_runtime";
    return Status(RET_INVALID_ARGUMENT, "Invalid schedule_output or llm_runtime");
  }

  // Set multi_batch_id
  result.schedule_output->multi_batch_id = multi_batch_id;

  llm_runtime_->ReorderInferRequests(result.schedule_output->running_reqs);

  // Create ForwardRequests
  result.grouped_reqs =
      std::make_shared<std::map<ModelInstance*, std::map<InferStage, std::vector<ForwardRequest*>>>>();
  llm_runtime_->BuildForwardRequests(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     *result.grouped_reqs);

  // Create SamplingRequests
  result.sampling_reqs = std::make_shared<std::vector<SamplingRequest>>();
  llm_runtime_->BuildSamplingRequest(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     *result.sampling_reqs);

  // No need for deep copy in sync mode, as there are no concurrent access issues
  // Note(qiannanzhou): In sync mode, hidden_token_num can be calculated later, but to unify the
  // SetHiddenUnitMeta call for both sync and async, it is calculated here
  size_t tokens = 0;
  for (size_t i = 0; i < result.schedule_output->running_reqs.size(); ++i) {
    tokens += result.schedule_output->running_reqs[i]->forwarding_tokens.size() -
              result.schedule_output->running_reqs[i]->kv_cached_token_num;
  }
  result.schedule_output->hidden_token_num = tokens;
  return Status();
}

void ScheduleProcessor::Start() {
  terminated_ = false;
  KLLM_LOG_INFO << "ScheduleProcessor started";
}

void ScheduleProcessor::Stop() {
  terminated_ = true;
  KLLM_LOG_INFO << "ScheduleProcessor stopped";
}

}  // namespace ksana_llm