/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/profiler/sched_event_tracer.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

BatchManager::BatchManager(const RuntimeConfig &runtime_config, std::shared_ptr<Context> context) {
  context_ = context;
  runtime_config_ = runtime_config;
}

Status BatchManager::RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance) {
  KLLM_LOG_DEBUG << "register model instance " << model_instance->name << " : " << model_instance.get();
  model_instances_[model_instance->name] = model_instance;
  return Status();
}

void BatchManager::SetBatchScheduler(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) {
  batch_scheduler_ = batch_scheduler;
}

void BatchManager::SetLlmRuntime(std::shared_ptr<LlmRuntime> llm_runtime) { llm_runtime_ = llm_runtime; }

void BatchManager::SetMultiBatchController(std::shared_ptr<MultiBatchController> controller) {
  multi_batch_controller_ = controller;
}

Status BatchManager::Enqueue(std::shared_ptr<Request> &req) {
  KLLM_LOG_DEBUG << "batch manager enqueue req id " << req->req_id;

  Status enqueue_status = Status(RetCode::RET_SUCCESS);

  if (model_instances_.find(req->model_name) == model_instances_.end()) {
    KLLM_LOG_ERROR << "req->model_name=" << req->model_name << " not found!";
    req->finish_status = Status(RET_INVALID_ARGUMENT, fmt::format("Model {} not found.", req->model_name));
    req->waiter->Notify();
    return req->finish_status;
  }
  const std::shared_ptr<ModelInstance> &model_instance = model_instances_[req->model_name];

  // Update `stop_token_ids` based on the config of the requested model.
  std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
  if (req->sampling_config.ignore_eos) {  // Ignore any end ids.
    stop_token_ids.clear();
  } else {  // Supplement the end ids in model config or generation config.
    for (int end_id : model_instance->GetModelConfig().end_ids) {
      if (std::find(stop_token_ids.begin(), stop_token_ids.end(), end_id) == stop_token_ids.end()) {
        stop_token_ids.push_back(end_id);
      }
    }
  }

  std::vector<std::shared_ptr<InferRequest>> infer_request_group;
  for (size_t i = 0; i < req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, i);
    infer_request_group.push_back(infer_req);
    infer_req->kv_cache_blocks.resize(runtime_config_.parallel_basic_config.attn_tensor_parallel_size);
    infer_req->block_token_num = runtime_config_.attn_backend_config.block_token_num;
    infer_req->model_instance = model_instance;
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;
    infer_req->step = 0;
    infer_req->kv_cached_token_num = 0;
  }

  for (auto &infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
  }

  enqueue_status = batch_scheduler_->AddInferRequest(infer_request_group);
  if (enqueue_status.OK()) {
    KLLM_LOG_DEBUG << "batch scheduler: added req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size() << " input tokens";
  } else {
    KLLM_LOG_ERROR << "batch scheduler: add req id " << req->req_id << " and "
                   << infer_request_group[0]->input_tokens.size()
                   << " input tokens failed, message: " << enqueue_status.ToString();
    if (req->sampling_config.num_beams > 1) {
      for (auto &infer_req : infer_request_group) {
        infer_req->ClearReqGroup();
      }
    }
  }

  return enqueue_status;
}

Status BatchManager::WaitAllDone() { return Status(); }

Status BatchManager::MainProcess(size_t multi_batch_id) {
  // Get block related information from device 0.
  // All devices have the same number of blocks.
  SetDevice(0);
  static time_t last_end_time_ms = ProfileTimer::GetCurrentTimeInMs();
  while (!terminated_) {
    time_t sched_start_time_ns = ProfileTimer::GetCurrentTimeInNs();
    std::shared_ptr<ScheduleOutputGroup> schedule_output_group = batch_scheduler_->Schedule(multi_batch_id);

    ScheduleOutput schedule_output = MergeScheduleOutputGroup(schedule_output_group);
    if (schedule_output_group->RunningSize() == 0) {
      multi_batch_controller_->NotifyCurrentBatchThreadNotReady(multi_batch_id);
      if (batch_scheduler_->IsIdle(multi_batch_id) && !terminated_) {
        batch_scheduler_->WaitUntilHaveReqs(multi_batch_id);
      } else {
        KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " not idle, sleep 100ms";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;
    }
    multi_batch_controller_->NotifyCurrentBatchIsStandby(multi_batch_id);
    RecordRequestSchedEventsWithStartTime(schedule_output.running_reqs, 0, multi_batch_id, 0, "Schedule",
                                          sched_start_time_ns);
    size_t forwarding_token_num = 0, total_seq_len = 0;
    for (auto &req : schedule_output.running_reqs) {
      forwarding_token_num += req->forwarding_tokens.size() - req->kv_cached_token_num;
      total_seq_len += req->forwarding_tokens.size();
    }

    schedule_output.multi_batch_id = multi_batch_id;
    time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();

    // Send schedule result to all workers if in distributed mode and init hidden unit buffer.
    if (!context_->IsStandalone()) {
      PROFILE_EVENT_SCOPE(LockAndBroadcastScheduleOutput, "LockAndBroadcastScheduleOutput");
      KLLM_LOG_MAIN << "wait to run multi_batch_id=" << multi_batch_id << ", epilogue=false";
      multi_batch_controller_->WaitUntilCurrentBatchCanRun(multi_batch_id);
      BroadcastScheduleOutput(&schedule_output);
      InitHiddenUnits(schedule_output.multi_batch_id);
    }
    RecordRequestSchedEvents(schedule_output.running_reqs, 0, schedule_output.multi_batch_id, 0, "PrepareForwarding",
                             RequestEventPhase::Begin);
    Status status = llm_runtime_->Step(&schedule_output, false);
    if (!status.OK()) {
      KLLM_LOG_ERROR << status.ToString();
    }

    time_t middle_time_ms = ProfileTimer::GetCurrentTimeInMs();

    // Wait until last worker done.
    if (!context_->IsStandalone()) {
      PROFILE_EVENT_SCOPE(SendAndStepOnChief_,
                          fmt::format("SendAndStepOnChief_{}_true", schedule_output.multi_batch_id));
      multi_batch_controller_->NotifyLastBatchHiddenUnitCanRecv(schedule_output.multi_batch_id);
      SendHiddenUnits(schedule_output.multi_batch_id);

      // lm head & sampling
      RecordRequestSchedEvents(schedule_output.running_reqs, 0, schedule_output.multi_batch_id, 0, "PrepareForwarding",
                               RequestEventPhase::Begin);
      status = llm_runtime_->Step(&schedule_output, true);
      if (!status.OK()) {
        KLLM_LOG_ERROR << status.ToString();
      }
      // free again.
      FreeHiddenUnits(schedule_output.multi_batch_id);
    }

    time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
    int global_token_throughput =
        (end_time_ms - last_end_time_ms) > 0 ? forwarding_token_num * 1000 / (end_time_ms - last_end_time_ms) : -1;
    int local_token_throuphput = forwarding_token_num * 1000 / (end_time_ms - start_time_ms);
    KLLM_LOG_MAIN << "multi_batch_id=" << multi_batch_id
                   << ", running_reqs.size=" << schedule_output.running_reqs.size()
                   << ", forwarding_token_num=" << forwarding_token_num << ", total_seq_len=" << total_seq_len
                   << ", 1st step " << (middle_time_ms - start_time_ms) << "ms, 2nd step "
                   << (end_time_ms - middle_time_ms) << "ms, total " << (end_time_ms - start_time_ms)
                   << "ms, local token throughput(tokens/s): " << local_token_throuphput
                   << ", global token throughput(tokens/s): " << global_token_throughput;
    last_end_time_ms = end_time_ms;

    REPORT_METRIC("global_token_throughput", global_token_throughput);
    REPORT_METRIC("local_token_throughput", local_token_throuphput);
    REPORT_METRIC("forwarding_token_num", forwarding_token_num);
    REPORT_METRIC("1st_step_time_ms", middle_time_ms - start_time_ms);
    REPORT_METRIC("2nd_step_time_ms", end_time_ms - middle_time_ms);
    REPORT_METRIC("total_step_time_ms", end_time_ms - start_time_ms);
  }

  return Status();
}

Status BatchManager::WorkerProcess() {
  SetDevice(0);
  while (!terminated_) {
    KLLM_LOG_MAIN << "Wait schedule_output from upstream node.";
    ScheduleOutput *schedule_output = GetScheduleOutputPool()->GetFromRecvQueue();
    if (!schedule_output) {
      break;
    }
    KLLM_LOG_MAIN << "WorkerProcess: start process schedule_output multi_batch_id=" << schedule_output->multi_batch_id;
    InitHiddenUnits(schedule_output->multi_batch_id);

    time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
    Status status = llm_runtime_->Step(schedule_output, false);
    if (!status.OK()) {
      KLLM_LOG_ERROR << status.ToString();
    }
    time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
    KLLM_LOG_MAIN << "multi_batch_id=" << schedule_output->multi_batch_id
                   << ", runningSize=" << schedule_output->running_reqs.size() << ", step cost "
                   << (end_time_ms - start_time_ms) << "ms";

    // Send hidden units to downstream node.
    SendHiddenUnits(schedule_output->multi_batch_id);

    // Free schedule output and hidden_unit..
    KLLM_LOG_DEBUG << "Free schedule output and hidden_unit.";
    GetScheduleOutputPool()->FreeScheduleOutput(schedule_output);

    // Free hidden units.
    FreeHiddenUnits(schedule_output->multi_batch_id);
  }

  return Status();
}

Status BatchManager::Start() {
  // Start main threads for standalone or master node of distributed mode.
  if (context_->IsChief()) {
    main_threads_.reserve(runtime_config_.max_pp_batch_num);
    for (size_t multi_batch_id = 0; multi_batch_id < runtime_config_.max_pp_batch_num; ++multi_batch_id) {
      main_threads_.push_back(
          std::unique_ptr<std::thread>(new std::thread(&BatchManager::MainProcess, this, multi_batch_id)));
    }
  } else {
    // Start worker thread only if in distributed mode, for worker node only.
    worker_thread_ = std::unique_ptr<std::thread>(new std::thread(&BatchManager::WorkerProcess, this));
  }

  return Status();
}

Status BatchManager::Stop() {
  KLLM_LOG_INFO << "Stop batch manager.";

  terminated_ = true;

  if (batch_scheduler_) {
    batch_scheduler_->Stop();
  }

  // stop data hub pool, will unlock the blocking Get().
  KLLM_LOG_INFO << "Stop data hub pool.";
  GetScheduleOutputPool()->Stop();
  if (!context_->IsStandalone()) {
    GetHiddenUnitBufferPool()->Stop();
  }

  KLLM_LOG_INFO << "Stop work thread.";
  if (context_->IsChief()) {
    for (auto &thread : main_threads_) {
      if (thread && thread->joinable()) {
        thread->join();
      }
    }
    main_threads_.clear();
  } else {
    if (worker_thread_ && worker_thread_->joinable()) {
      worker_thread_->join();
    }
  }

  // Clear model intances.
  model_instances_.clear();

  KLLM_LOG_INFO << "batch manager stopped.";
  return Status();
}

ScheduleOutput BatchManager::MergeScheduleOutputGroup(std::shared_ptr<ScheduleOutputGroup> &schedule_output_group) {
  ScheduleOutput merged_schedule_output;
  if (schedule_output_group->outputs.size() == 1) {
    merged_schedule_output = *(schedule_output_group->outputs.at(0));
    merged_schedule_output.multi_batch_id = schedule_output_group->schedule_id;
  } else if (schedule_output_group->outputs.size() == 0) {
    return merged_schedule_output;
  } else {
    // NOTE(karlluo): merge all schedule group outputs into one schedule output instance
    merged_schedule_output.multi_batch_id = schedule_output_group->schedule_id;

    // schedule_output_group->outputs.size() equal to the number of attention data parallelism size
    merged_schedule_output.finish_req_ids.resize(schedule_output_group->outputs.size());
    merged_schedule_output.merged_swapout_req_ids.resize(schedule_output_group->outputs.size());
    merged_schedule_output.merged_swapin_req_ids.resize(schedule_output_group->outputs.size());
    merged_schedule_output.swapout_req_block_ids.resize(schedule_output_group->outputs.size());
    merged_schedule_output.swapin_req_block_ids.resize(schedule_output_group->outputs.size());

    // NOTE(karlluo): calculate reserve vector space
    size_t running_reqs_reserve_size = 0;
    size_t worker_running_reqs_reserve_size = 0;
    for (size_t attn_dp_idx = 0; attn_dp_idx < schedule_output_group->outputs.size(); ++attn_dp_idx) {
      ScheduleOutput *schedule_output = schedule_output_group->outputs.at(attn_dp_idx);
      if (schedule_output == nullptr) {
        continue;
      }

      if (schedule_output->finish_req_ids.size() >= 1) {
        merged_schedule_output.finish_req_ids[attn_dp_idx] = schedule_output->finish_req_ids[0];
      }
      if (schedule_output->merged_swapout_req_ids.size() >= 1) {
        merged_schedule_output.merged_swapout_req_ids[attn_dp_idx] = schedule_output->merged_swapout_req_ids[0];
      }
      if (schedule_output->merged_swapin_req_ids.size() >= 1) {
        merged_schedule_output.merged_swapin_req_ids[attn_dp_idx] = schedule_output->merged_swapin_req_ids[0];
      }
      if (schedule_output->swapout_req_block_ids.size() >= 1) {
        merged_schedule_output.swapout_req_block_ids[attn_dp_idx] = schedule_output->swapout_req_block_ids[0];
      }
      if (schedule_output->swapin_req_block_ids.size() >= 1) {
        merged_schedule_output.swapin_req_block_ids[attn_dp_idx] = schedule_output->swapin_req_block_ids[0];
      }

      running_reqs_reserve_size += schedule_output->running_reqs.size();
      worker_running_reqs_reserve_size += schedule_output->worker_running_reqs.size();
    }

    merged_schedule_output.running_reqs.reserve(running_reqs_reserve_size);
    merged_schedule_output.worker_running_reqs.reserve(worker_running_reqs_reserve_size);

    for (size_t attn_dp_idx = 0; attn_dp_idx < schedule_output_group->outputs.size(); ++attn_dp_idx) {
      ScheduleOutput *schedule_output = schedule_output_group->outputs.at(attn_dp_idx);
      if (schedule_output == nullptr) {
        continue;
      }

      for (auto req = schedule_output->running_reqs.begin(); req != schedule_output->running_reqs.end(); ++req) {
        (*req)->attn_dp_group_id = attn_dp_idx;
      }
      for (auto req = schedule_output->worker_running_reqs.begin(); req != schedule_output->worker_running_reqs.end();
           ++req) {
        (*req)->attn_dp_group_id = attn_dp_idx;
      }

      merged_schedule_output.running_reqs.insert(merged_schedule_output.running_reqs.end(),
                                                 schedule_output->running_reqs.begin(),
                                                 schedule_output->running_reqs.end());
      merged_schedule_output.worker_running_reqs.insert(merged_schedule_output.worker_running_reqs.end(),
                                                        schedule_output->worker_running_reqs.begin(),
                                                        schedule_output->worker_running_reqs.end());
    }
  }

  return merged_schedule_output;
}

}  // namespace ksana_llm
