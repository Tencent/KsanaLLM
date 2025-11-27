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
      async_sched_threads_.emplace_back(&ScheduleProcessor::AsyncScheduleThread, this, multi_batch_id);
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

  std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
  KLLM_CHECK(
      batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs));
  ProcessScheduleDataInternal(multi_batch_id, *result);
  return result;
}

void ScheduleProcessor::UpdateWithGenerationResult(size_t multi_batch_id,
                                                   const GenerationOutputGroup& generation_output) {
  batch_scheduler_->UpdateWithGenerationResult(multi_batch_id, generation_output);

  // Async mode, check if there are any launchable schedule results
  if (enable_async_) {
    std::lock_guard<std::mutex> lock(planning_result_mutex_);
    if (planning_sched_results_[multi_batch_id] != nullptr) {
      std::shared_ptr<ScheduleResult> result = planning_sched_results_[multi_batch_id];
      std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
      KLLM_CHECK(
          batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs));
      if (!result->schedule_output->running_reqs.empty()) {
        KLLM_LOG_SCHEDULER << "Put result running_reqs size=" << result->schedule_output->running_reqs.size();
        ProcessScheduleDataInternal(multi_batch_id, *result);
        sched_result_queue_[multi_batch_id].Put(result);
      } else {
        KLLM_LOG_SCHEDULER << "Drop result because running_reqs is empty";
      }
      for (const auto& req : stopped_reqs) {
        RemoveRequestFromQueue(result->generation_output_group.reqs[req->attn_dp_group_id], req);
      }
      planning_sched_results_[multi_batch_id] = nullptr;
      // Notify waiting AsyncScheduleThread that the planning result has been consumed
      planning_result_cv_.notify_one();
    }
  }
}

void ScheduleProcessor::AsyncScheduleThread(size_t multi_batch_id) {
  while (!terminated_) {
    // Check if there's existing planning result that needs to be consumed first
    {
      std::unique_lock<std::mutex> lock(planning_result_mutex_);
      // Wait until planning_sched_results_[multi_batch_id] is nullptr or thread is terminated
      planning_result_cv_.wait(
          lock, [this, multi_batch_id]() { return planning_sched_results_[multi_batch_id] == nullptr || terminated_; });
    }

    if (terminated_) {
      break;
    }

    std::shared_ptr<ScheduleResult> result = Schedule(multi_batch_id);
    if (terminated_) {
      sched_result_queue_[multi_batch_id].Put(nullptr);
      return;
    }

    std::vector<std::shared_ptr<InferRequest>> stopped_reqs;
    if (batch_scheduler_->TryToLaunchPlannedScheduleOutput(multi_batch_id, *(result->schedule_output), stopped_reqs)) {
      if (!result->schedule_output->running_reqs.empty()) {
        KLLM_LOG_SCHEDULER << "Put result running_reqs size=" << result->schedule_output->running_reqs.size();
        ProcessScheduleDataInternal(multi_batch_id, *result);
        sched_result_queue_[multi_batch_id].Put(result);
      } else {
        KLLM_LOG_SCHEDULER << "Drop result because running_reqs is empty";
      }
    } else {
      if (result->schedule_output->running_reqs.size() > 0) {
        std::lock_guard<std::mutex> lock(planning_result_mutex_);
        assert(planning_sched_results_[multi_batch_id] == nullptr);
        planning_sched_results_[multi_batch_id] = result;
      }
    }
    for (const auto& req : stopped_reqs) {
      RemoveRequestFromQueue(result->generation_output_group.reqs[req->attn_dp_group_id], req);
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
      NotifyCurrentBatchThreadNotReady(multi_batch_id);
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
  result->outputs = schedule_output_group->outputs;

  // 3. Merge schedule results
  result->schedule_output = std::make_shared<ScheduleOutput>();
  MergeScheduleOutputGroupRunningRequests(schedule_output_group, *(result->schedule_output));

  // 4. There are running requests, process the data
  result->generation_output_group.BuildFromScheduleOutputGroup(*schedule_output_group);

  //  schedule output in schedule_output_group have been used, clear running_reqs to avoid blocking schedule.
  for (auto& scheduled_out : schedule_output_group->outputs) {
    scheduled_out->ClearRunningReqs();
  }
  return result;
}

void ScheduleProcessor::NotifyCurrentBatchThreadNotReady(size_t multi_batch_id) {
  multi_batch_controller_->NotifyCurrentBatchThreadNotReady(multi_batch_id);
}

Status ScheduleProcessor::ProcessScheduleDataInternal(size_t multi_batch_id, ScheduleResult& result) {
  if (!result.schedule_output || !llm_runtime_) {
    KLLM_LOG_ERROR << "Invalid schedule_output or llm_runtime";
    return Status(RET_INVALID_ARGUMENT, "Invalid schedule_output or llm_runtime");
  }

  // Set multi_batch_id
  result.schedule_output->multi_batch_id = multi_batch_id;

  // Disable scheduler to avoid changing InferRequest members.
  batch_scheduler_->Lock();
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
  batch_scheduler_->Unlock();

  result.schedule_output->hidden_token_num = tokens;
  return Status();
}

void ScheduleProcessor::Stop() {
  terminated_ = true;

  // Notify all waiting threads to wake up and check terminated_ flag
  planning_result_cv_.notify_all();

  // Stop async threads
  for (auto& thread : async_sched_threads_) {
    thread.join();
  }
  KLLM_LOG_INFO << "ScheduleProcessor stopped";
}

}  // namespace ksana_llm
