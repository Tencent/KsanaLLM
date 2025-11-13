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
  if (enable_async_) {
    planning_sched_results_.resize(max_pp_batch_num_);
    for (size_t multi_batch_id = 0; multi_batch_id < max_pp_batch_num_; ++multi_batch_id) {
      async_sched_threads_.push_back(
          std::unique_ptr<std::thread>(new std::thread(&ScheduleProcessor::AsyncScheduleThread, this, multi_batch_id)));
    }
  }

  KLLM_LOG_INFO << "ScheduleProcessor initialized";
}

// The sync mode: call Schedule directly -> check for running requests -> wait or process data
std::shared_ptr<ScheduleResult> ScheduleProcessor::GetNextScheduleResult(size_t multi_batch_id) {
  if (enable_async_) {
    return sched_result_queue_[multi_batch_id].Get();
  }
  // Sync mode
  std::shared_ptr<ScheduleResult> result = Schedule(multi_batch_id);
  if (terminated_) {
    return nullptr;
  }
  KLLM_CHECK(result->schedule_output->IsLaunchable());
  ProcessLaunchableScheduleResult(multi_batch_id, result);

  return result;
}

void ScheduleProcessor::UpdateWithGenerationResult(size_t multi_batch_id,
                                                   const GenerationOutputGroup& generation_output) {
  batch_scheduler_->UpdateWithGenerationResult(multi_batch_id, generation_output);

  // Async mode, check if there are any launchable schedule results
  if (enable_async_) {
    std::shared_ptr<ScheduleResult> result;
    {
      std::lock_guard<std::mutex> lock(planning_result_mutex);
      if (planning_sched_results_[multi_batch_id] != nullptr) {
        assert(planning_sched_results_[multi_batch_id]->schedule_output->IsLaunchable());
        result = planning_sched_results_[multi_batch_id];
        planning_sched_results_[multi_batch_id] = nullptr;
      }
    }
    if (result) {
      ProcessLaunchableScheduleResult(multi_batch_id, result);
      sched_result_queue_[multi_batch_id].Put(result);
    }
  }
}

void ScheduleProcessor::AsyncScheduleThread(size_t multi_batch_id) {
  while (!terminated_) {
    std::shared_ptr<ScheduleResult> result = Schedule(multi_batch_id);
    if (terminated_) {
      sched_result_queue_[multi_batch_id].Put(nullptr);
      return;
    }
    if (result->schedule_output->IsLaunchable()) {
      ProcessLaunchableScheduleResult(multi_batch_id, result);
      sched_result_queue_[multi_batch_id].Put(result);
    } else {
      std::lock_guard<std::mutex> lock(planning_result_mutex);
      assert(planning_sched_results_[multi_batch_id] == nullptr);
      planning_sched_results_[multi_batch_id] = result;
    }
  }
}

std::shared_ptr<ScheduleResult> ScheduleProcessor::Schedule(size_t multi_batch_id) {
  std::shared_ptr<ScheduleResult> result;
  std::shared_ptr<ScheduleOutputGroup> schedule_output_group;
  while (!terminated_) {
    // 1. Call Schedule directly
    schedule_output_group = batch_scheduler_->Schedule(multi_batch_id);

    // 2. Check if there are any running requests
    if (schedule_output_group->RunningSize() == 0) {
      // No running requests, need to wait
      // TODO(robertyuan): NotifyCurrentBatchThreadNotReady will block this thread with inflight tasks
      multi_batch_controller_->NotifyCurrentBatchThreadNotReady(multi_batch_id);
      if (batch_scheduler_->IsIdle(multi_batch_id)) {
        batch_scheduler_->WaitUntilHaveReqs(multi_batch_id);
      } else {
        KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " not idle, sleep 100ms";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;  // Continue the loop to wait
    }
    break;
  }

  if (terminated_) {
    return nullptr;
  }

  result = std::make_shared<ScheduleResult>();

  // 3. Merge schedule results
  result->schedule_output = std::make_shared<ScheduleOutput>();
  MergeScheduleOutputGroup(schedule_output_group, *(result->schedule_output));

  // 4. There are running requests, process the data
  result->generation_output_group.BuildFromScheduleOutputGroup(*schedule_output_group);
  result->outputs = schedule_output_group->outputs;
  return result;
}

void ScheduleProcessor::ProcessLaunchableScheduleResult(size_t multi_batch_id, std::shared_ptr<ScheduleResult> result) {
  // Remove finished requests in async mode
  for (auto it = result->schedule_output->running_reqs.begin(); it != result->schedule_output->running_reqs.end();) {
    auto req = *it;
    if (req->finished) {
      it = result->schedule_output->running_reqs.erase(it);
    } else {
      ++it;
    }
  }

  // Launch running_reqs in schedule output
  for (auto& scheduled_out : result->outputs) {
    KLLM_CHECK(scheduled_out->IsLaunchable());
    scheduled_out->LaunchScheduleOutput();
    scheduled_out->Clear();  // Information have been copied to result.schedule_output, clear it
  }

  // Todo(robertyuan): Seperate ForwardRequest and SamplingRequest building logic to accelerate in async mode
  ProcessScheduleDataInternal(multi_batch_id, *result);
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
  llm_runtime_->BuildForwardRequests(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     result.grouped_reqs);

  // Create SamplingRequests
  llm_runtime_->BuildSamplingRequest(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     result.sampling_reqs);

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

void ScheduleProcessor::Stop() {
  terminated_ = true;

  // Stop async threads
  if (enable_async_) {
    for (auto& thread : async_sched_threads_) {
      if (thread->joinable()) {
        thread->join();
      }
    }
  }
  KLLM_LOG_INFO << "ScheduleProcessor stopped";
}

}  // namespace ksana_llm
