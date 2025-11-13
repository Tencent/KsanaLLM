/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
#include <cmath>
#include <memory>

#include "base_strategy.h"
#include "ksana_llm/batch_scheduler/state/scheduler_tick_tok.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/stop_checker.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

bool RemoveRequestFromQueue(std::vector<std::shared_ptr<InferRequest>> &req_queue,
                            const std::shared_ptr<InferRequest> &req) {
  auto it = std::find(req_queue.begin(), req_queue.end(), req);
  if (it != req_queue.end()) {
    KLLM_CHECK((*it)->req_id == req->req_id);
    req_queue.erase(it);
    return true;
  }
  return false;
}

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config,
                                                       const RuntimeConfig &runtime_config)
    : BaseScheduleStrategy(batch_scheduler_config, runtime_config) {
  const auto env = Singleton<Environment>::GetInstance();
  env->GetConnectorConfigs(connector_config_);
  if (connector_config_.group_role != GroupRole::NONE) {
    TransferEngine::GetInstance()->Initialize(connector_config_.group_role);
    if (batch_scheduler_config_.split_fuse_token_num > 0) {
      KLLM_LOG_INFO << "Disable split fuse in prefill-decoding separating node";
      batch_scheduler_config_.split_fuse_token_num = 0;
    }
  }

  /* TODO(zezhao):
   * 在多机 EP 场景下，每台机器都持有完整的 MLA、Embedding 以及 LmHead，多台机器间仅在 MOE 层进行数据共享
   * 对于每台机器，MLA 部分的所有 DP 节点，每轮调度后会产出最多 max_step_token_num 的 token。
   * 多台机器通过 DeepEP Dispatch 逻辑，完成 AllToAll 数据传输，则每台机器、每张卡上理论收到的最多 token 数为：
   *    machine_nums * max_step_token_num
   * 而 MOE 部分所使用的参与数据存储的几个空间 hidden_buffer_0, hidden_buffer_1, reduce_buffer, workspace 等，
   * 均是按照 max_step_token_num 分配的显存空间。上述的 Dispatch 分发会导致计算越界。
   * 因此这里暂时通过将 dp_max_step_token_num 缩放到 (1 / EP机器数) 的方法，规避越界问题。
   * 后续将重新调整 MOE 部分的空间分配及使用方法，移除此处的缩放操作。
   * 额外的，由于缩放操作存在，在开启双机 EP 时，将 max_step_token_num 配置为 64K，则程序本身仅能支持最大为 32K 的
   * 请求，与 yaml 配置存在不符。
   */
  dp_max_step_token_num_ =
      batch_scheduler_config_.max_step_token_num / runtime_config_.parallel_basic_config.expert_world_size;
  dp_max_batch_size_ = batch_scheduler_config_.max_batch_size;
  dp_max_logits_num_ = dp_max_batch_size_ * batch_scheduler_config.max_decode_tokens_per_req;
  if (connector_config_.group_role == GroupRole::DECODE) {
    dp_max_decode_batch_size_ = dp_max_batch_size_;
    // 增加预参数的大小
    dp_max_batch_size_ = batch_scheduler_config_.max_batch_size + (batch_scheduler_config_.max_pretransfer_batch_size);
    KLLM_LOG_INFO << "decode dp_max_batch_size_:" << dp_max_batch_size_
                  << ", dp_max_decode_batch_size_:" << dp_max_decode_batch_size_;
  }
}

size_t ContinuousBatchingStrategy::GetMaxRequiredTokenNum(const size_t token_num) const {
  return token_num + runtime_config_.mtp_step_num - std::min(runtime_config_.mtp_step_num, 1ul);
}

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >=
         req->timestamp_in_us / 1000 + batch_scheduler_config_.waiting_timeout_in_ms;
}

Status ContinuousBatchingStrategy::RecomputeMockRequest(std::shared_ptr<InferRequest> &req) {
  KLLM_LOG_SCHEDULER << "RecomputeMockRequest " << req;

  // Add request to the beginning of waiting queue.
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  req->kv_cache_blocks.assign(runtime_config.parallel_basic_config.attn_tensor_parallel_size, {});
  req->RebuildBlockPtrs();
  req->ResetPrefillingTokens();

  // To avoid Mock requests being categorized as SingleTokenForward requests, we
  // calculate the Mock request total length as: Mock total length = MTP token
  // count + SingleToken length + 1 (additional token)
  const size_t mock_request_length = runtime_config.mtp_step_num + 1 + 1;
  // After Mock request completes one inference round, rollback the newly generated tokens at the end to restore
  // the initial state.
  if (req->output_tokens.size() > mock_request_length) {
    req->output_tokens.resize(mock_request_length);
  }
  if (req->GetPlanningSequenceLen() > mock_request_length) {
    req->forwarding_tokens.resize(mock_request_length);
  }
  batch_state_->mock_queue.push_back(req);
  return Status();
}

bool ContinuousBatchingStrategy::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
#ifdef CLEAR_CACHE
  // For testing purposes only
  // When a specific input is received, free all cached blocks.
  // TODO(robertyuan): Remove this to right place
  if (req->input_tokens.size() == 2 && req->input_tokens[0] == 0 && req->input_tokens[1] == 0) {
    size_t free_block_num;
    auto prefix_cache_manager_ptr = dynamic_cast<PrefixCacheManager *>(cache_manager_.get());
    if (prefix_cache_manager_ptr != nullptr) {
      prefix_cache_manager_ptr->FreeCachedBlocks(1e8, free_block_num);
      KLLM_LOG_WARNING << "cache_manager free " << free_block_num << " blocks.";
    }
  }
#endif
  if (req->IsEosGenerated() ||
      (req->sampling_config.max_new_tokens > 0 &&
       req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
      req->output_tokens.size() >= batch_scheduler_config_.max_token_len) {
    stop_checker_->CheckCompleteStopStrings(req);
    KLLM_LOG_SCHEDULER << "Request " << req->req_id << " had finished."
                       << " req output_tokens size: " << req->output_tokens.size()
                       << " input_tokens size: " << req->input_tokens.size();
    return true;
  }
  // TODO(robertyuan): User defined checking. Move this to PostGenHook
  // When stop strings are checked and matched, stop early
  return stop_checker_->CheckIncrementalStopStrings(req);
}
// TODO(robertyuan): Move to PostScheduleHook
void ContinuousBatchingStrategy::DetermineDraftNum(std::shared_ptr<InferRequest> req) {
  // Determine the number of draft_tokens to generate in the current step based on the scheduling status.
  req->suggested_draft_num = 0;
  constexpr size_t kDraftBatchSizeThreshold = 16;
  const size_t running_bs = batch_state_->decoding_queue.size();
  if (running_bs >= kDraftBatchSizeThreshold) {
    return;
  }
  const size_t draft_num_per_req = (kDraftBatchSizeThreshold - running_bs) / running_bs;
  req->suggested_draft_num = draft_num_per_req;
}

void ContinuousBatchingStrategy::RecomputeRequest(std::shared_ptr<InferRequest> req) {
  if (!req->HasInflightTask()) {
    // No task is running, recompute immediately
    SyncRecomputeRequest(req);
  } else {
    batch_state_->async_recomputed_reqs[req->req_id] = req;
  }
}

void ContinuousBatchingStrategy::SyncRecomputeRequest(std::shared_ptr<InferRequest> req) {
  KLLM_LOG_SCHEDULER << "SyncRecomputeRequest req id is: " << req->req_id
                     << " kv_comm_request_id: " << req->kv_comm_request_id;

  // Add request to the beginning of waiting queue.
  req->kv_cache_blocks.assign(runtime_config_.parallel_basic_config.attn_tensor_parallel_size, {});
  req->checksummed_block_num.assign(runtime_config_.parallel_basic_config.attn_tensor_parallel_size, 0);
  req->ResetPrefillingTokens();

  if (connector_config_.group_role != GroupRole::NONE) {
    KLLM_LOG_INFO << "Request " << req->req_id << "  and kv_comm_request_id: " << req->kv_comm_request_id
                  << " is recomputed due to exceeding max_step_token_num or max_batch_size in decode group.";
    Status status(RET_PREDICTOR_DISCARD, "Disaggregation of prefill and decoding could not be recomputed.");
    ResetRequest(req, status, true);
    return;
  }

  static constexpr bool terminate = false;
  ResetRequest(req, Status(RET_SUCCESS, "RecomputeRequest"), terminate);

  batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), req);
}

void ContinuousBatchingStrategy::StopRequest(std::shared_ptr<InferRequest> req, Status ret_status,
                                             RequestState req_state) {
  KLLM_LOG_SCHEDULER << "StopRequest req id is: " << req->req_id;
  if (req->HasInflightTask()) {
    batch_state_->async_stoped_reqs[req->req_id] = {req, ret_status, req_state};
  } else {
    SyncStopRequest(req, ret_status, req_state);
  }
}

void ContinuousBatchingStrategy::SyncStopRequest(std::shared_ptr<InferRequest> req, Status ret_status,
                                                 RequestState req_state) {
  ResetRequest(req, ret_status, true);
  // Record finish req_id
  if (req_state == RequestState::REQUEST_STATE_RUNNING) {
    if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
      size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
      for (size_t idx = 0; idx < needed_push_size; ++idx) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
      }
    }
    batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
  }
}

bool ContinuousBatchingStrategy::ProcessAsyncStoppedRequest(std::shared_ptr<InferRequest> &req) {
  auto it = batch_state_->async_stoped_reqs.find(req->req_id);
  if (it == batch_state_->async_stoped_reqs.end()) {
    return false;
  }
  const auto &stop_info = it->second;

  KLLM_CHECK(!req->HasInflightTask());

  KLLM_LOG_SCHEDULER << "Processing async stop request " << req->req_id;
  SyncStopRequest(req, stop_info.ret_status, stop_info.req_state);
  batch_state_->async_stoped_reqs.erase(it);
  return true;
}

bool ContinuousBatchingStrategy::ProcessAsyncRecomputeRequest(std::shared_ptr<InferRequest> &req) {
  auto it = batch_state_->async_recomputed_reqs.find(req->req_id);
  if (it == batch_state_->async_recomputed_reqs.end()) {
    return false;
  }

  KLLM_CHECK(!req->HasInflightTask());
  KLLM_LOG_SCHEDULER << "Processing async recompute request " << req->req_id;
  SyncRecomputeRequest(req);

  batch_state_->async_recomputed_reqs.erase(it);
  return true;
}

// In asynchronous mode, inflighting requests may be in other queues.
void ContinuousBatchingStrategy::RemoveRequestFromBatchState(const std::shared_ptr<InferRequest> &req) {
  if (RemoveRequestFromQueue(batch_state_->schedule_output->running_reqs, req)) {
    // TODO(robertyuan): Found in running queue, adjust planning schedule output if needed.
  }
  if (RemoveRequestFromQueue(batch_state_->decoding_queue, req) ||
      RemoveRequestFromQueue(batch_state_->waiting_queue, req)) {
    return;
  }
  KLLM_LOG_FATAL << "Request " << req->req_id << " not found in batch state";
  // TODO(robertyuan): request maybe moved to another batch state.
}

void ContinuousBatchingStrategy::EstimateRequestPlanningWorkload(std::shared_ptr<InferRequest> req) {
  if (req->HasInflightTask()) {
    // Estimate task generation result
    // TODO(robertyuan): replace with GenerationResultEstimator
    if (req->GetInflightTask().workload.sampling_token_num > 0) {
      size_t max_draft_token_num = 2;
      req->SetInflightTaskGenResultEstimation(kStepGenerateTokenNum, max_draft_token_num);
    }
  }

  req->SetPlanningWorkload(req->GetRemainingWorkload());
}

void ContinuousBatchingStrategy::ResetRequest(std::shared_ptr<InferRequest> req, Status ret_status, bool terminate) {
  KLLM_LOG_SCHEDULER << "ResetRequest " << req->ScheduleStateToStr() << ", ret_status:" << ret_status.ToString()
                     << ", terminate:" << terminate;

  req->finish_status = req_status;
  req->finished = terminate;
  req->RebuildBlockPtrs();

  cache_manager_->DestroyFinishedRequest(req->req_id);

  if (terminate) {
    req->Notify();
  }
}

std::pair<size_t, size_t> ContinuousBatchingStrategy::CheckRunningQueueStepTokens(
    const std::vector<std::shared_ptr<InferRequest>> &checking_reqs,
    std::vector<std::shared_ptr<InferRequest>> &passed_reqs) {
  // step_token_num: Controls the maximum total tokens in a batch (max_step_token_num), related to buffer allocation and
  // GPU memory management
  // step_not_kv_cached_token_num: Total count of tokens without KV caching, directly affecting computational workload
  // requirements
  passed_reqs.reserve(checking_reqs.size());

  scheduler_ticktok_->Barrier();

  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reset();
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  scheduler_ticktok_->Barrier();

  size_t step_token_num = 0, step_not_kv_cached_token_num = 0;
  size_t total_sampling_token_num = 0, req_num = 0;

  // count how many tokens can be scheduled in this step
  // the req_token_num is the snapshot when the request is added to running queue.
  // it may be smaller than the actual token number when the request is running.
  size_t local_step_token_num = 0;
  for (auto it = checking_reqs.begin(); it != checking_reqs.end();) {
    const auto &req = *it;
    // TODO(robertyuan): replace with step_token_num. They should be same in all case, check later
    const size_t not_kv_cached_token_num = req->GetPlanningQueryLen();
    // In Forward(), attention kernel use buffer with all tokens when step_token_num> threshold.
    // TODO(robertyuan): Adjust this after kernel problem fixed.
    const size_t req_token_num = not_kv_cached_token_num <= GetDecodeTokenNumThreshold()
                                     ? not_kv_cached_token_num
                                     : req->GetPlanningSequenceLen();

    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    req_num = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    total_sampling_token_num = scheduler_shared_counter_->step_logits_num.Get();

    if (step_token_num + req_token_num > dp_max_step_token_num_ ||
        total_sampling_token_num + req->sampling_token_num > dp_max_logits_num_ || req_num >= dp_max_batch_size_) {
      scheduler_ticktok_->Unlock();
      continue;
    }
    step_not_kv_cached_token_num += not_kv_cached_token_num;

    scheduler_shared_counter_->step_batch_size.Increase(1);
    scheduler_shared_counter_->step_token_num.Increase(req_token_num);
    scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
    local_step_token_num += req_token_num;
    passed_reqs.push_back(req);
    scheduler_ticktok_->Unlock();

    ++it;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();

  return {local_step_token_num, step_not_kv_cached_token_num};
}

void ContinuousBatchingStrategy::UpdateRunningRequests(const std::vector<std::shared_ptr<InferRequest>> &running_reqs) {
  KLLM_LOG_SCHEDULER << "update running requests size:" << running_reqs.size();

  for (auto req : running_reqs) {
    // All req here should be decode now.
    req->infer_stage = InferStage::kDecode;
    req->UpdateAfterInflightTaskFinished();
    req->ResetInflightTask();
    // clear flexible cache copy tasks after context stage is finished
    req->flexible_cached_copy_tasks.clear();

    ReportRequestProgressInfo(req);

    if (ProcessAsyncStoppedRequest(req)) {
      continue;
    }
    if (ProcessAsyncRecomputeRequest(req)) {
      continue;
    }

    req->req_ctx->emplace("status_code", std::to_string(static_cast<int>(req->finish_status.GetCode())));

    // Always update cache manager, even if request is finished.
    Status status = cache_manager_->UpdateRequestTokens(req->req_id, req->forwarding_tokens, req->kv_cached_token_num,
                                                        req->kv_cache_blocks);

    if (!status.OK()) {
      KLLM_LOG_SCHEDULER << "UpdateRequestTokens " << req << " error, recompute it, info: " << status.GetMessage();
      RemoveRequestFromBatchState(req);
      RecomputeRequest(req);
      continue;
    }

    // TODO(zakwang): PD support StructuredOutput
    // If prefill finished, enter transfer queue
    if (connector_config_.group_role == GroupRole::PREFILL && !req->is_mock_req) {
      req->NotifyStep();
      KLLM_LOG_SCHEDULER << "Prefill enter transfer queue for tranfer task to Decode, req id:"
                         << req->kv_comm_request_id;
      RemoveRequestFromBatchState(req);
      batch_state_->transfer_queue.emplace_back(req);
      continue;
    }

    // Check if finished.
    // ProcessPrefillTransferQueue also checks if the request is finished.
    if (CheckRequestFinish(req)) {
      const auto end_time = ProfileTimer::GetCurrentTimeInUs();
      const size_t output_token_num = req->output_tokens.size() - req->input_tokens.size();
      const uint64_t duration = end_time - req->timestamp_in_us;
      if (req->finish_status.GetCode() == RET_SUCCESS && output_token_num > 0) {
        REPORT_METRIC("total_latency_us", duration);
        REPORT_METRIC("output_token_len", output_token_num);
        REPORT_METRIC("input_token_len", req->input_tokens.size());
      } else {
        REPORT_METRIC("forward_req_error_num", req->finish_status.GetCode());
      }

      StopRequest(req, Status(RET_SUCCESS), RequestState::REQUEST_STATE_RUNNING);

      // Put mock request back to mock_queue.
      if (req->is_mock_req) {
        RecomputeMockRequest(req);
      }
      RemoveRequestFromBatchState(req);
      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      REPORT_COUNTER("forward_req_timeout_num", static_cast<size_t>(1));
      KLLM_LOG_ERROR << "req timeout in running:" << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in running."), RequestState::REQUEST_STATE_RUNNING);
      RemoveRequestFromBatchState(req);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_WARNING << "req aborted in running: " << req->req_id;

      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in running."), RequestState::REQUEST_STATE_RUNNING);
      RemoveRequestFromBatchState(req);
      REPORT_COUNTER("forward_req_aborted_num", static_cast<size_t>(1));
      continue;
    }

    // Not finished, notify streaming iterator.
    req->NotifyStep();

    KLLM_LOG_SCHEDULER << "schedule_state=" << req->ScheduleStateToStr()
                       << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                       << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
  }
}

void ContinuousBatchingStrategy::ProcessDecodingQueue() {
  PROFILE_EVENT_SCOPE(ProcessDecodingQueue, "ProcessDecodingQueue");
  KLLM_LOG_SCHEDULER << "ProcessDecodingQueue invoked: " << *batch_state_
                     << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                     << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  // Estimate inflight task result and update planning workloads
  for (auto &req : batch_state_->decoding_queue) {
    EstimateRequestPlanningWorkload(req);
  }

  // Check decoding queue to determine whether it exceeds the max_step_token_num.
  std::vector<std::shared_ptr<InferRequest>> passed_reqs;
  CheckRunningQueueStepTokens(batch_state_->decoding_queue, passed_reqs);

  if (passed_reqs.size() == 0) {
    return;
  }

  size_t alloc_failed_req_num = 0;
  for (auto req : passed_reqs) {
    KLLM_CHECK(req->GetPlanningWorkload().prefill_token_num == 0);  // Decoding requests only
    const size_t max_required_token_num = GetMaxRequiredTokenNum(req->GetPlanningSequenceLen());
    size_t req_step_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, max_required_token_num);
    size_t forwarded_token_num = req->GetPlanningSequenceLen() - req->GetPlanningQueryLen();

    Status status = cache_manager_->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks);
    if (status.OK()) {
      DetermineDraftNum(req);
      batch_state_->schedule_output->running_reqs.push_back(req);
      continue;
    }

    KLLM_LOG_ERROR << "Alllocate blocks error, info: " << status.GetMessage();
    alloc_failed_req_num++;
  }

  if (alloc_failed_req_num == 0) {
    return;
  }

  // No more blocks, skip waiting_req launch.
  batch_state_->step_sched_finish = true;
  KLLM_LOG_SCHEDULER << "No more free blocks for " << alloc_failed_req_num << " decoding reqs, skip waiting queue."
                     << *batch_state_;

  if (batch_state_->schedule_output->running_reqs.size() > 0) {
    return;
  }

  // If no decoding request can run due to block limitation, disable split fuse to free some blocks
  size_t recomputing_waiting_task_num = 0;
  if (!batch_state_->waiting_queue.empty() && batch_scheduler_config_.split_fuse_token_num > 0) {
    std::vector<std::shared_ptr<InferRequest>> recomputing_reqs;
    for (auto &waiting_req : batch_state_->waiting_queue) {
      if (!waiting_req->kv_cache_blocks[0].empty()) {
        recomputing_reqs.push_back(waiting_req);
      }
    }
    KLLM_LOG_WARNING << fmt::format(
        "Split fuse disabled due to allocation failure. {} chunked prefilling requests will be recomputed",
        recomputing_reqs.size());
    recomputing_waiting_task_num = recomputing_reqs.size();
    for (auto req : recomputing_reqs) {
      RemoveRequestFromQueue(batch_state_->waiting_queue, req);
      RecomputeRequest(req);
    }
    batch_scheduler_config_.split_fuse_token_num = 0;
    KLLM_LOG_WARNING << "Split fuse has been disabled.";
  }

  KLLM_LOG_SCHEDULER << "recomputing_waiting_task_num=" << recomputing_waiting_task_num
                     << ", async_stoped_reqs.size()=" << batch_state_->async_stoped_reqs.size()
                     << ", async_recomputed_reqs.size()=" << batch_state_->async_recomputed_reqs.size();
  // If waiting queue cannot release any blocks and no asyn recopmute/stop tasks, try to recompute some decoding tasks;
  if ((recomputing_waiting_task_num == 0) && (batch_state_->async_stoped_reqs.size() == 0) &&
      (batch_state_->async_recomputed_reqs.size() == 0)) {
    size_t recompute_min_cost = std::numeric_limits<size_t>::max();
    std::shared_ptr<InferRequest> recompute_min_cost_req;
    for (auto req : passed_reqs) {
      size_t forwarded_token_num = req->GetPlanningSequenceLen() - req->GetPlanningQueryLen();
      if (recompute_min_cost > forwarded_token_num) {
        recompute_min_cost = forwarded_token_num;
        recompute_min_cost_req = req;
      }
    }
    RemoveRequestFromQueue(batch_state_->decoding_queue, recompute_min_cost_req);
    RecomputeRequest(recompute_min_cost_req);
  }
}

void ContinuousBatchingStrategy::ProcessWaitingQueue() {
  PROFILE_EVENT_SCOPE(ProcessWaitingQueue, "ProcessWaitingQueue");
  KLLM_LOG_SCHEDULER << "ProcessWaitingQueue invoked, waiting queue size:" << batch_state_->waiting_queue.size()
                     << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                     << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  const size_t decode_request_num = batch_state_->schedule_output->running_reqs.size();
  std::vector<std::shared_ptr<InferRequest>> passed_reqs;
  auto [step_token_num, step_not_kv_cached_token_num] =
      CheckRunningQueueStepTokens(batch_state_->schedule_output->running_reqs, passed_reqs);
  KLLM_CHECK(passed_reqs.size() == batch_state_->schedule_output->running_reqs.size());

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size() + batch_state_->transfer_queue.size();
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->GetPlanningSamplingTokenNum();
  }

  scheduler_ticktok_->Barrier();

  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reset();
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  scheduler_ticktok_->Barrier();

  scheduler_shared_counter_->step_batch_size.Increase(step_batch_size);
  scheduler_shared_counter_->step_token_num.Increase(step_token_num);
  scheduler_shared_counter_->step_logits_num.Increase(step_logits_num);

  // Make sure all dp groups are accumulated.
  scheduler_ticktok_->Barrier();

  if (batch_state_->waiting_queue.empty()) {
    scheduler_ticktok_->Skip();
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_SCHEDULER << "Skip processing waiting_queue." << *batch_state_;
    scheduler_ticktok_->Skip();
    return;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    req->cache_manager = cache_manager_;
    ++it;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    EstimateRequestPlanningWorkload(req);

    // Check timeout, no finished req in waiting queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_SCHEDULER << "req timeout in waiting:" << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in waiting."), false);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_SCHEDULER << "req aborted in waiting:" << req;
      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in waiting."), RequestState::REQUEST_STATE_WAITING);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    bool is_last_prefill_step = true;
    size_t req_step_block_num = 0;  // block number to be allocated in this step
    size_t shared_token_num = 0;
    size_t seq_token_num = req->GetPlanningSequenceLen();  // Trying to process remaining prefilling tokens
    ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
    if (planning_workload.prefill_start_offset == 0) {
      // Find shared blocks in first step
      size_t shared_block_num = 0;
      cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->GetPrefillingTokens(), seq_token_num,
                                                  shared_block_num, req_step_block_num, shared_token_num);
      if (shared_token_num > 0) {
        REPORT_COUNTER("prefix_cache_hit_req_num", static_cast<size_t>(1));
        REPORT_COUNTER("prefix_cache_hit_token_num", shared_token_num);
        REPORT_COUNTER("prefix_cache_hit_block_num", shared_block_num);
        KLLM_LOG_SCHEDULER << "req_id=" << req->req_id << " find shared_block_num=" << shared_block_num
                           << ", shared_token_num=" << shared_token_num;
      }
    } else {
      // Last task is chunked prefill task, previous tokens are sharable
      shared_token_num = planning_workload.prefill_start_offset;
    }

    // If splitfuse is enabled, and token number to be processed is larger than split_fuse_token_num,
    // adjust planning workload
    size_t req_step_token_num = seq_token_num - shared_token_num;
    if ((batch_scheduler_config_.split_fuse_token_num > 0) &&
        (req_step_token_num > batch_scheduler_config_.split_fuse_token_num)) {
      // not the last prefill step, adjust planning workload
      KLLM_LOG_SCHEDULER << "Not the last prefill step, req_step_token_num:" << req_step_token_num
                         << ", split_fuse_token_num:" << batch_scheduler_config_.split_fuse_token_num;

      is_last_prefill_step = false;
      planning_workload.prefill_token_num -= req_step_token_num - batch_scheduler_config_.split_fuse_token_num;
      req_step_token_num = batch_scheduler_config_.split_fuse_token_num;

      req->SetPlanningWorkload(planning_workload);
      seq_token_num = req->GetPlanningSequenceLen();

      assert(seq_token_num == shared_token_num + req_step_token_num);
      assert(!planning_workload.IsEmpty());
    }

    req_step_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, seq_token_num);
    KLLM_LOG_SCHEDULER << "req_step_token_num:" << req_step_token_num << ", req_step_block_num:" << req_step_block_num
                       << ", seq_token_num:" << seq_token_num;

    if (is_last_prefill_step) {
      // add extra token for MTP sub model if needed
      req_step_block_num += (runtime_config_.mtp_step_num + req->block_token_num - 1) / req->block_token_num;
    }

    const size_t launch_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.launch_block_threshold);

    // In this context, all other db groups will be paused.
    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    step_batch_size = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    step_logits_num = scheduler_shared_counter_->step_logits_num.Get();

    // Get usable block number every time, the blocks matched by req are not reusable here.
    const size_t total_free_block_num = cache_manager_->GetRequestUsableBlockNumber(req->req_id);
    if (step_logits_num + req->sampling_token_num <= dp_max_logits_num_ && step_batch_size < dp_max_batch_size_ &&
        step_token_num + seq_token_num <= dp_max_step_token_num_ &&
        req_step_block_num + launch_block_threshold <= total_free_block_num) {
      // Assume we could succ, so that the AllocateRequestBlocks is not blocked.
      scheduler_shared_counter_->step_batch_size.Increase(1);
      scheduler_shared_counter_->step_token_num.Increase(seq_token_num);
      scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
      scheduler_ticktok_->Unlock();

      Status status = cache_manager_->AllocateRequestBlocks(req->req_id, req_step_block_num, req->kv_cache_blocks);
      if (status.OK()) {
        step_not_kv_cached_token_num += seq_token_num - shared_token_num;
        req->RebuildBlockPtrs();

        // if full matched, skip decode and put it to the end of decode list.
        if (shared_token_num == seq_token_num) {
          KLLM_LOG_SCHEDULER << "Full matched, skip prefill, " << req->ScheduleStateToStr();
          REPORT_COUNTER("full_prompt_matched_req_num", static_cast<size_t>(1));
          // If shared_token_num is equal to forwarding size, means all tokens have valid kv cache
          // but don't have logits to generate next token, need to recompute the last token to generate next token
          // NOTE: this happens only when request is just received.

          req->infer_stage = InferStage::kDecode;
          planning_workload.prefill_start_offset = shared_token_num - kStepGenerateTokenNum;
          planning_workload.prefill_token_num = kStepGenerateTokenNum;
          req->SetPlanningWorkload(planning_workload);
          req->SetRemainingWorkload(planning_workload);
        } else {
          KLLM_LOG_SCHEDULER << "shared token not equal forwarding size, " << req->ScheduleStateToStr();
          assert(seq_token_num > shared_token_num);
          if (shared_token_num > 0) {
            size_t planning_prefill_token_num =
                planning_workload.prefill_token_num + planning_workload.prefill_start_offset;
            planning_workload.prefill_start_offset = shared_token_num;
            planning_workload.prefill_token_num = planning_prefill_token_num - shared_token_num;
            req->SetPlanningWorkload(planning_workload);

            ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
            size_t prefill_token_num = remaining_workload.prefill_token_num + remaining_workload.prefill_start_offset;
            remaining_workload.prefill_start_offset = shared_token_num;
            remaining_workload.prefill_token_num = prefill_token_num - shared_token_num;
            req->SetRemainingWorkload(remaining_workload);
          }
          if (connector_config_.group_role == GroupRole::DECODE) {
            batch_state_->transfer_queue.emplace_back(req);
            it = batch_state_->waiting_queue.erase(it);
            KLLM_LOG_SCHEDULER << "Decode put req to transfer queue, req id: " << req->kv_comm_request_id;
            continue;
          }
        }

        batch_state_->schedule_output->running_reqs.push_back(req);
        if (is_last_prefill_step) {
          batch_state_->decoding_queue.push_back(req);
          it = batch_state_->waiting_queue.erase(it);
        } else {
          it++;
        }

        // The flexible cache handling could be placed prior to the split_fuse break. Try moving it after testing.
        cache_manager_->UpdateFlexibleCache(req->req_id, req->forwarding_tokens, shared_token_num,
                                            req->flexible_cached_copy_tasks);
        continue;
      } else {
        // decrease counter if failed, thread-safe, no timing-control needed.
        scheduler_shared_counter_->step_batch_size.Decrease(1);
        scheduler_shared_counter_->step_token_num.Decrease(seq_token_num);
        scheduler_shared_counter_->step_logits_num.Decrease(req->sampling_token_num);

        KLLM_LOG_ERROR << "Allocate blocks error, waiting req can not be launched, " << req->ScheduleStateToStr()
                       << ", msg=" << status.GetMessage();
      }
    } else {
      scheduler_ticktok_->Unlock();
    }

    KLLM_LOG_SCHEDULER << "total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                       << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
    break;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();
}

void ContinuousBatchingStrategy::ProcessTransferQueue() {
  PROFILE_EVENT_SCOPE(ProcessTransferQueue, "ProcessTransferQueue");
  if (connector_config_.group_role == GroupRole::DECODE) {
    ProcessDecodeTransferQueue();
    AddTransferMeta(batch_state_->transfer_queue);
  }
  if (connector_config_.group_role == GroupRole::PREFILL) {
    ProcessPrefillTransferQueue();
    AddTransferMeta(batch_state_->schedule_output->running_reqs);
  }
}

void ContinuousBatchingStrategy::Schedule(std::vector<std::shared_ptr<InferRequest>> &waiting_reqs) {
  batch_state_->ResetInfoBeforeSchedule();
  scheduler_ticktok_->SetThreadIndex(dp_group_id_);
  if (dp_group_id_ == 0) {
    scheduler_shared_counter_->step_batch_size.Reset(0);
    scheduler_shared_counter_->step_token_num.Reset(0);
    scheduler_shared_counter_->step_logits_num.Reset(0);
  }

  if (connector_config_.group_role != GroupRole::NONE) {
    // 对waiting_reqs排序，kv_comm_request_id小的在前
    std::sort(waiting_reqs.begin(), waiting_reqs.end(),
              [](const auto &a, const auto &b) { return a->kv_comm_request_id < b->kv_comm_request_id; });
  }
  batch_state_->MergeWaitingReqs(waiting_reqs);
  auto start_us = ProfileTimer::GetCurrentTimeInUs();

  // if last schedule output is not launched
  if (batch_state_->schedule_output->running_reqs.size() > 0) {
    return;
  }

  ProcessDecodingQueue();
  auto running_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_running_queue_time_us", running_queue_end_us - start_us);
  ProcessWaitingQueue();
  auto waiting_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_waiting_queue_time_us", waiting_queue_end_us - running_queue_end_us);
  ProcessTransferQueue();
  auto transfer_queue_end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_transfer_queue_time_us", transfer_queue_end_us - waiting_queue_end_us);

  // Must barrier before reorder.
  scheduler_ticktok_->Barrier();

  // Change next visit order of dp groups, for load balance.
  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reorder();
  }

  batch_state_->schedule_output->SetPlanningTask();

  auto end_us = ProfileTimer::GetCurrentTimeInUs();
  REPORT_METRIC("batch_scheduler_time_us", end_us - start_us);
}

void ContinuousBatchingStrategy::ReportRequestProgressInfo(const std::shared_ptr<InferRequest> req) {
  if (req->kv_cached_token_num > req->computed_token_num && !req->is_mock_req) {
    if (connector_config_.group_role == GroupRole::DECODE && req->computed_token_num == 0) {
      req->computed_token_num = req->input_tokens.size();
    }
    auto computed_token_num = req->kv_cached_token_num - req->computed_token_num;
    if (req->kv_cached_token_num < req->input_tokens.size()) {
      REPORT_COUNTER("computed_input_token_num", computed_token_num);
    }
    if (req->kv_cached_token_num == req->input_tokens.size()) {
      REPORT_COUNTER("computed_input_token_num", computed_token_num);
      REPORT_COUNTER("computed_output_token_num", 1);
      REPORT_COUNTER("computed_token_num", 1);
    }
    if (req->kv_cached_token_num > req->input_tokens.size()) {
      REPORT_COUNTER("computed_output_token_num", computed_token_num);
    }
    REPORT_COUNTER("computed_token_num", computed_token_num);
    req->computed_token_num = req->kv_cached_token_num;
  }
}

}  // namespace ksana_llm
