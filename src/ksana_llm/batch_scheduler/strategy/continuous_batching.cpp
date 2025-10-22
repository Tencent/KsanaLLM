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

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config,
                                                       const RuntimeConfig &runtime_config)
    : BaseScheduleStrategy(batch_scheduler_config, runtime_config) {
  const auto env = Singleton<Environment>::GetInstance();

  env->GetConnectorConfigs(connector_config_);
  if (connector_config_.group_role != GroupRole::NONE) {
    TransferEngine::GetInstance()->Initialize(connector_config_.group_role);
  }

  const size_t attn_data_parallel_size = runtime_config_.parallel_basic_config.attn_data_parallel_size;
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
    dp_max_batch_size_ = batch_scheduler_config_.max_batch_size +
                         (batch_scheduler_config_.max_pretransfer_batch_size);
    KLLM_LOG_INFO << "decode dp_max_batch_size_:" << dp_max_batch_size_
                   << ", dp_max_decode_batch_size_:" << dp_max_decode_batch_size_;
  }
}

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >= req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms;
}

Status ContinuousBatchingStrategy::RecomputeMockRequest(std::shared_ptr<InferRequest> &req, bool is_swap_req) {
  KLLM_LOG_DEBUG << "RecomputeMockRequest " << req;

  // Add request to the beginning of waiting queue.
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  req->kv_cache_blocks.assign(runtime_config.parallel_basic_config.attn_tensor_parallel_size, {});
  req->RebuildBlockPtrs();
  req->infer_stage = InferStage::kContext;
  req->step = 0;
  req->kv_cached_token_num = 0;
  req->suggested_draft_num = 0;
  req->prefix_cache_len = 0;
  // To avoid Mock requests being categorized as SingleTokenForward requests, we calculate the Mock request total
  // length as: Mock total length = MTP token count + SingleToken length + 1 (additional token)
  size_t mock_request_length = (runtime_config.enable_mtp_module ? 1 : 0) + 1 + 1;
  // After Mock request completes one inference round, rollback the newly generated tokens at the end to restore
  // the initial state.
  if (req->output_tokens.size() > mock_request_length) {
    req->output_tokens.resize(mock_request_length);
  }
  if (req->forwarding_tokens.size() > mock_request_length) {
    req->forwarding_tokens.resize(mock_request_length);
  }
  batch_state_->mock_queue.push_back(req);
  return Status();
}

bool ContinuousBatchingStrategy::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
  std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
#ifdef CLEAR_CACHE
  if (req->input_tokens.size() == 2 && req->input_tokens[0] == 0 && req->input_tokens[1] == 0) {
    size_t free_block_num;
    auto prefix_cache_manager_ptr = dynamic_cast<PrefixCacheManager *>(cache_manager_.get());
    if (prefix_cache_manager_ptr != nullptr) {
      prefix_cache_manager_ptr->FreeCachedBlocks(1e8, free_block_num);
      KLLM_LOG_WARNING << "cache_manager free " << free_block_num << " blocks.";
    }
  }
#endif
  if (std::find(stop_token_ids.begin(), stop_token_ids.end(), req->output_tokens.back()) != stop_token_ids.end() ||
      (req->sampling_config.max_new_tokens > 0 &&
       req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
      req->output_tokens.size() >= batch_scheduler_config_.max_token_len) {
    stop_checker_->CheckCompleteStopStrings(req);
    KLLM_LOG_DEBUG << "Request " << req->req_id << " had finished."
                   << " req output_tokens size: " << req->output_tokens.size()
                   << " input_tokens size: " << req->input_tokens.size();
    return true;
  }

  // When stop strings are checked and matched, stop early
  return stop_checker_->CheckIncrementalStopStrings(req);
}

void ContinuousBatchingStrategy::DetermineDraftNum(std::shared_ptr<InferRequest> req) {
  // Determine the number of draft_tokens to generate in the current step based on the scheduling status.
  req->suggested_draft_num = 0;
  constexpr size_t kDraftBatchSizeThreshold = 16;
  const size_t running_bs = batch_state_->schedule_output->running_reqs.size();
  if (running_bs >= kDraftBatchSizeThreshold) {
    return;
  }
  const size_t draft_num_per_req = (kDraftBatchSizeThreshold - running_bs) / running_bs;
  req->suggested_draft_num = draft_num_per_req;
}

std::vector<std::shared_ptr<InferRequest>>::iterator ContinuousBatchingStrategy::RecomputeRequest(
    std::vector<std::shared_ptr<InferRequest>>::iterator &it, bool is_swap_req) {
  auto req = *it;
  HandleRecomputeRequest(req, is_swap_req);
  return batch_state_->schedule_output->running_reqs.erase(it);
}

void ContinuousBatchingStrategy::HandleRecomputeRequest(std::shared_ptr<InferRequest> req, bool is_swap_req) {
  KLLM_LOG_DEBUG << "HandleRecomputeRequest req id is: " << req->req_id
                 << " kv_comm_request_id: " << req->kv_comm_request_id;

  // Add request to the begining of waiting queue.
  req->kv_cache_blocks.assign(runtime_config_.parallel_basic_config.attn_tensor_parallel_size, {});
  req->checksummed_block_num.assign(runtime_config_.parallel_basic_config.attn_tensor_parallel_size, 0);
  req->infer_stage = InferStage::kContext;
  req->step = 0;
  req->kv_cached_token_num = 0;
  req->suggested_draft_num = 0;
  req->prefix_cache_len = 0;

  if (connector_config_.group_role != GroupRole::NONE) {
    KLLM_LOG_INFO << "Request " << req->req_id << "  and kv_comm_request_id: " << req->kv_comm_request_id
                  << " is recomputed due to exceeding max_step_token_num or max_batch_size in decode group.";
    Status status(RET_PREDICTOR_DISCARD, "Disaggregation of prefill and decoding could not be recomputed.");
    ResetRequest(req, status, is_swap_req, true);
    return;
  }

  static constexpr bool terminate = false;
  ResetRequest(req, Status(RET_SUCCESS, "RecomputeRequest"), is_swap_req, terminate);

  batch_state_->waiting_queue.emplace_front(req);
}

void ContinuousBatchingStrategy::StopRequest(std::shared_ptr<InferRequest> req, Status ret_status,
                                             RequestState req_state) {
  const bool is_swap_req = (req_state == RequestState::REQUEST_STATE_SWAPPED);
  ResetRequest(req, ret_status, is_swap_req, true);
  // Record finish req_id
  if (req_state == RequestState::REQUEST_STATE_RUNNING || req_state == RequestState::REQUEST_STATE_SWAPPED) {
    if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
      size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
      for (size_t idx = 0; idx < needed_push_size; ++idx) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
      }
    }
    batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
  }
}

void ContinuousBatchingStrategy::AsyncStopRequest(std::shared_ptr<InferRequest> req, Status req_status,
                                                  bool is_swap_req) {
  KLLM_LOG_DEBUG << "AsyncStopRequest " << *req << ", ret_status:" << ret_status.ToString()
                 << ", is_swapped_req:" << is_swap_req << ", delay finish and notify";

  // 设置完成状态但不设置 finished=true，也不调用 Notify()
  req->finish_status = ret_status;
  // 将请求放入延迟处理队列，等待在推理完成之后处理。这是为了避免在推理过程中释放请求，
  // 导致 Request 对象在请求处理（Step）尚未完成时就被销毁，从而引发了对已释放成员变量的非法访问。
  async_destroyed_reqs_.push_back(req);

  // Record finish req_id
  if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
    size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
    for (size_t idx = 0; idx < needed_push_size; ++idx) {
      batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
    }
  }
  batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
}

std::vector<std::shared_ptr<InferRequest>>::iterator ContinuousBatchingStrategy::AsyncRecomputeRequest(
    std::vector<std::shared_ptr<InferRequest>>::iterator &it, bool is_swap_req) {
  auto req = *it;
  KLLM_LOG_DEBUG << "AsyncRecomputeRequest " << *req << ", is_swapped_req:" << is_swap_req << ", delay recompute";

  async_recomputed_reqs_.push_back({req, is_swap_req});
  return batch_state_->schedule_output->running_reqs.erase(it);
}

void ContinuousBatchingStrategy::NotifyAsyncFinishedRequests() {
  // 处理延迟的请求完成通知
  for (auto &req : async_destroyed_reqs_) {
    req->finished = true;
    req->Notify();
    cache_manager_->DestroyFinishedRequest(req->req_id);
    KLLM_LOG_DEBUG << "Async request " << req->req_id << " completion notification sent";
  }
  async_destroyed_reqs_.clear();
}

void ContinuousBatchingStrategy::NotifyAsyncRecomputedRequests() {
  for (auto [req, is_swap_req] : async_recomputed_reqs_) {
    KLLM_LOG_DEBUG << "Async recompute request " << req->req_id << " notification sent";
    HandleRecomputeRequest(req, is_swap_req);
  }
  async_recomputed_reqs_.clear();
}

void ContinuousBatchingStrategy::ResetRequest(std::shared_ptr<InferRequest> req, Status ret_status, bool is_swap_req,
                                              bool terminate) {
  KLLM_LOG_DEBUG << "ResetRequest " << *req << ", req_status:" << req_status.ToString()
                 << ", is_swap_req:" << is_swap_req << ", terminate:" << terminate;

  req->finish_status = req_status;
  req->finished = terminate;
  req->RebuildBlockPtrs();

  if (is_swap_req) {
    cache_manager_->DestroySwappedRequest(req->req_id);
  } else {
    cache_manager_->DestroyFinishedRequest(req->req_id);
  }

  if (terminate) {
    req->Notify();
  }
}

std::pair<size_t, size_t> ContinuousBatchingStrategy::CheckRunningQueueStepTokens() {
  // step_token_num: Controls the maximum total tokens in a batch (max_step_token_num), related to buffer allocation and
  // GPU memory management
  // step_not_kv_cached_token_num: Total count of tokens without KV caching, directly affecting computational workload
  // requirements

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
  for (auto it = batch_state_->schedule_output->running_reqs.begin();
       it != batch_state_->schedule_output->running_reqs.end();) {
    const auto &req = *it;
    const size_t not_kv_cached_token_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
    const size_t req_token_num = not_kv_cached_token_num <= decode_token_num_threshold_ ? not_kv_cached_token_num
                                                                                        : req->forwarding_tokens.size();

    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    req_num = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    total_sampling_token_num = scheduler_shared_counter_->step_logits_num.Get();

    if (step_token_num + req_token_num > dp_max_step_token_num_ ||
        total_sampling_token_num + req->sampling_token_num > dp_max_logits_num_ || req_num >= dp_max_batch_size_) {
      scheduler_ticktok_->Unlock();
      if (batch_scheduler_config_.enable_async) {
        it = AsyncRecomputeRequest(it, false);
      } else {
        it = RecomputeRequest(it, false);
      }
      continue;
    }
    step_not_kv_cached_token_num += not_kv_cached_token_num;

    scheduler_shared_counter_->step_batch_size.Increase(1);
    scheduler_shared_counter_->step_token_num.Increase(req_token_num);
    scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
    local_step_token_num += req_token_num;
    scheduler_ticktok_->Unlock();

    ++it;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();

  return {local_step_token_num, step_not_kv_cached_token_num};
}

void ContinuousBatchingStrategy::UpdateRunningRequests() {
  batch_state_->ResetInfoBeforeSchedule();
  KLLM_LOG_DEBUG << "update running requests size:" << batch_state_->schedule_output->running_reqs.size();
  for (auto it = batch_state_->schedule_output->running_reqs.begin();
       it != batch_state_->schedule_output->running_reqs.end();) {
    auto req = *it;
    // All req here should be decode now.
    req->infer_stage = InferStage::STATE_DECODE;

    // remove rejected draft token
    // TODO(david): for mtp, it should consider step. The accepted token num equal mtp times.
    if (!batch_scheduler_config_.enable_async) {
      req->forwarding_tokens.resize(req->forwarding_tokens.size() - req->forwarding_tokens_draft_num +
                                    req->accepted_tokens.size());
    }
    // current token has kv_cache
    req->kv_cached_token_num = req->forwarding_tokens.size();
    req->prefix_cache_len = req->kv_cached_token_num;
    // clear flexible cache copy tasks after context stage is finished
    req->flexible_cached_copy_tasks.clear();

    // Always update cache manager, even if request is finished.
    // TODO(david): for mtp, it should consider fake prefix.
    // If the fake prefix block will be removed for the LRU algorithm.
    Status status = cache_manager_->UpdateRequestTokens(req->req_id, req->forwarding_tokens, req->kv_cached_token_num,
                                                        req->kv_cache_blocks);

    //  if it is async, req->output_tokens.size() should minus 1
    // TODO(qiannan) 需要check为什么需要-1
    int real_output_num = req->output_tokens.size();
    if (batch_scheduler_config_.enable_async) {
      real_output_num -= 1;
    }
    if (req->forwarding_tokens.size() - req->accepted_tokens.size() < real_output_num) {
      // When the request actually needs to calculate Multi-Token, insert the request into the waiting queue.
      // This is primarily used for the split-fuse feature.
      req->infer_stage = InferStage::STAGE_CONTEXT;
      req->forwarding_tokens = req->output_tokens;
      req->draft_tokens.clear();
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      req->sampling_token_num = kStepGenerateTokenNum;
      req->last_step_token_num = kStepGenerateTokenNum;
      batch_state_->waiting_queue.emplace_front(req);
      it = batch_state_->schedule_output->running_reqs.erase(it);
      KLLM_LOG_DEBUG << "splitfuse cal multi-token" << *req;
      continue;
    }

    // append generated token
    // for async, the generated_token is faked
    req->forwarding_tokens.emplace_back(req->generated_token);
    req->sampling_token_num = kStepGenerateTokenNum;

    // append new tokens to output_tokens
    // for async, it should append after forward finishing
    if (!batch_scheduler_config_.enable_async) {
      req->output_mutex.lock();
      req->output_tokens.insert(req->output_tokens.end(),
                                req->forwarding_tokens.end() - req->accepted_tokens.size() - kStepGenerateTokenNum,
                                req->forwarding_tokens.end());
      req->output_mutex.unlock();
    }
    // TODO(david): for mtp, req->accepted_tokens.size() should fake
    req->last_step_token_num = req->accepted_tokens.size() + kStepGenerateTokenNum;

    req->req_ctx->emplace("status_code", std::to_string(static_cast<int>(req->finish_status.GetCode())));

    if (!status.OK()) {
      KLLM_LOG_ERROR << "UpdateRequestTokens " << req << " error, recompute it, info: " << status.GetMessage();
      req->draft_tokens.clear();
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      if (batch_scheduler_config_.enable_async) {
        it = AsyncRecomputeRequest(it, false);
      } else {
        it = RecomputeRequest(it, false);
      }
      continue;
    }

    // TODO(zakwang): PD support StructuredOutput
    if (connector_config_.group_role == GroupRole::PREFILL && !req->is_mock_req) {
      req->NotifyStep();
      KLLM_LOG_DEBUG << "Prefill enter transfer queue for tranfer task to Decode, req id:" << req->kv_comm_request_id;
      batch_state_->transfer_queue.emplace_back(req);
      it = batch_state_->schedule_output->running_reqs.erase(it);
      continue;
    }

    // Check if finished.
    // ProcessPrefillTransferQueue also checks if the request is finished.
    if (CheckRequestFinish(req)) {
      const auto end_time = ProfileTimer::GetCurrentTimeInMs();
      const size_t output_token_num = req->output_tokens.size() - req->input_tokens.size();
      const uint64_t duration = end_time - req->timestamp_in_ms;
      if (req->finish_status.GetCode() == RET_SUCCESS && output_token_num > 0) {
        REPORT_METRIC("total_latency_ms", duration);
        REPORT_METRIC("output_token_len", output_token_num);
        REPORT_METRIC("input_token_len", req->input_tokens.size());
      } else {
        REPORT_METRIC("forward_req_error_num", req->finish_status.GetCode());
      }

      // Record finish req_id
      if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
        size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
        for (size_t idx = 0; idx < needed_push_size; ++idx) {
          batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
        }
      }
      batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      // for async, the kvblock should be destroyed later to avoid the trash token forward error.
      if (batch_scheduler_config_.enable_async) {
        AsyncStopRequest(req, Status(RET_SUCCESS), false);
      } else {
        StopRequest(req, Status(RET_SUCCESS), RequestState::REQUEST_STATE_RUNNING);
      }

      // Put mock request back to mock_queue.
      if (req->is_mock_req) {
        RecomputeMockRequest(req, 0);
      }

      it = batch_state_->schedule_output->running_reqs.erase(it);

      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      REPORT_COUNTER("forward_req_timeout_num", static_cast<size_t>(1));
      KLLM_LOG_ERROR << "req timeout in running:" << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in running."), false);
      it = batch_state_->schedule_output->running_reqs.erase(it);

      // Record finish req_id
      if (req->attn_dp_group_id == batch_state_->schedule_output->finish_req_ids.size()) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_WARNING << "req aborted in running: " << req->req_id;

      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in running."), false);
      it = batch_state_->schedule_output->running_reqs.erase(it);

      // Record finish req_id
      if (req->attn_dp_group_id == batch_state_->schedule_output->finish_req_ids.size()) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      REPORT_COUNTER("forward_req_aborted_num", static_cast<size_t>(1));

      continue;
    }

    // Not finished, notify streaming iterator.
    req->NotifyStep();

    // append draft tokens
    // TODO(david): for mtp, the draft token should modify in step 0 to avoid exceeding max_seq_len
    const std::vector<int> &draft_tokens = req->draft_tokens.GetDraftTokens();
    // TODO(qiannan) 这里需要插入正确数量的draft fake token
    req->forwarding_tokens.insert(req->forwarding_tokens.end(), draft_tokens.begin(), draft_tokens.end());
    req->forwarding_tokens_draft_num = req->draft_tokens.size();
    req->sampling_token_num =
        req->logits_custom_length > 0 ? req->logits_custom_length : req->draft_tokens.size() + kStepGenerateTokenNum;

    KLLM_LOG_DEBUG << *req << "forwarding_tokens: " << req->forwarding_tokens
                   << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                   << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
    ++it;
  }
}

size_t ContinuousBatchingStrategy::GetRunningRequestsBlockNeed() {
  size_t total_needed_block_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    total_needed_block_num += cache_manager_->GetRequestStepBlockNumber(req->req_id, req->forwarding_tokens.size());
  }
  return total_needed_block_num;
}

Status ContinuousBatchingStrategy::AllocateRequestBlocksWithRetry(std::shared_ptr<InferRequest> req,
                                                                  size_t &total_needed_block_num,
                                                                  size_t &step_block_num, bool &allocate_block_succ,
                                                                  bool &skip_swapout_check) {
  Status status = cache_manager_->AllocateRequestBlocks(req->req_id, step_block_num, req->kv_cache_blocks);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Alllocate blocks error, info: " << status.GetMessage();
    MergePendingSwapoutRequests(true, false);

    // Try the allocation again after all swapout finished.
    status = cache_manager_->AllocateRequestBlocks(req->req_id, step_block_num, req->kv_cache_blocks);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Alllocate blocks error again, recompute it, info: " << status.GetMessage();
      allocate_block_succ = false;
    } else {
      total_needed_block_num -= step_block_num;
      skip_swapout_check = true;
    }
  } else {
    total_needed_block_num -= step_block_num;
    skip_swapout_check = true;
  }
  return status;
}

void ContinuousBatchingStrategy::ProcessRunningQueue() {
  PROFILE_EVENT_SCOPE(ProcessRunningQueue, "ProcessRunningQueue");
  KLLM_LOG_DEBUG << "ProcessRunningQueue invoked: " << *batch_state_
                 << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  // Merge pending swapin requests, continue running.
  Status status = MergePendingSwapinRequests(false, true);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "ProcessRunningQueue error, info: " << status.GetMessage();
  }

  size_t total_needed_block_num = GetRunningRequestsBlockNeed();

  if (batch_state_->schedule_output->running_reqs.empty() && batch_state_->waiting_queue.empty()) {
    // If running & waiting queue in current step is empty, wait all swapin jobs done if existed.
    // In order to make sure the schedule result not empty.
    if (!batch_state_->swapin_pending_requests_.empty()) {
      Status status = MergePendingSwapinRequests(true, false);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "MergePendingSwapinRequests error, info: " << status.GetMessage();
      }

      KLLM_LOG_DEBUG << "ProcessRunningQueue update, running queue size:"
                     << batch_state_->schedule_output->running_reqs.size()
                     << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                     << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
    }
  }

  // Check Running Queue to determine whether it exceeds the max_step_token_num.
  CheckRunningQueueStepTokens();

  // Swapout necessary blocks.
  bool skip_swapout_check = false;
  for (size_t running_batch_size = batch_state_->schedule_output->running_reqs.size(); running_batch_size > 0;
       --running_batch_size) {
    auto it = batch_state_->schedule_output->running_reqs.begin() + running_batch_size - 1;
    auto req = *it;

    // No need to check max_batch_size and max_step_token_num here.
    const size_t swapout_block_threshold =
        std::ceil(running_batch_size * batch_scheduler_config_.swapout_block_threshold);

    // Whether the allocation operation is successful.
    bool allocate_block_succ = true;

    size_t step_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, req->forwarding_tokens.size());
    size_t total_free_block_num = cache_manager_->GetUsableBlockNumber();
    size_t future_free_block_num = cache_manager_->GetFutureFreeBlockNumber();

    // If last request have not enough blocks, wait all swapout done.
    if (running_batch_size == 1 && step_block_num > total_free_block_num) {
      KLLM_LOG_DEBUG << "Not enough blocks for last " << req << ", waiting all pending swapout requests done.";
      MergePendingSwapoutRequests(true, false);

      // Update block num.
      total_free_block_num = cache_manager_->GetUsableBlockNumber();
      future_free_block_num = cache_manager_->GetFutureFreeBlockNumber();
    }

    // never swap out last request.
    if (skip_swapout_check ||
        (step_block_num <= total_free_block_num && total_needed_block_num <= total_free_block_num &&
         total_needed_block_num + swapout_block_threshold <= total_free_block_num + future_free_block_num)) {
      KLLM_LOG_DEBUG << "continue running " << *req << *batch_state_ << ", running_batch_size:" << running_batch_size
                     << ", skip_swapout_check:" << skip_swapout_check << ",step_block_num:" << step_block_num
                     << ", total_free_block_num:" << total_free_block_num
                     << ", total_needed_block_num:" << total_needed_block_num
                     << ", configed_swapout_block_threshold:" << batch_scheduler_config_.swapout_block_threshold
                     << ", swapout_block_threshold:" << swapout_block_threshold
                     << ", future_free_block_num:" << future_free_block_num
                     << ", allocate_block_succ:" << allocate_block_succ
                     << ", current_block_num:" << req->kv_cache_blocks[0].size();

      status = AllocateRequestBlocksWithRetry(req, total_needed_block_num, step_block_num, allocate_block_succ,
                                              skip_swapout_check);
      if (status.OK()) {
        DetermineDraftNum(req);
        continue;
      }
    }

    // If allocation failed, disable split fuse.
    if (!batch_state_->waiting_queue.empty() && batch_scheduler_config_.split_fuse_token_num > 0) {
      for (auto &req : batch_state_->waiting_queue) {
        if (!req->kv_cache_blocks[0].empty()) {
          // Note(TJ): maybe cloud use StopRequest
          cache_manager_->DestroyFinishedRequest(req->req_id);
          req->kv_cache_blocks.assign(runtime_config_.parallel_basic_config.attn_tensor_parallel_size, {});
          req->RebuildBlockPtrs();
          KLLM_LOG_WARNING << fmt::format("Split fuse disabled due to allocation failure for request ID {}",
                                          req->req_id);
        }
      }
      batch_scheduler_config_.split_fuse_token_num = 0;
      KLLM_LOG_WARNING << "Split fuse has been disabled.";
    }

    // If allocation failed, skip swapout, recompute request directly.
    if (allocate_block_succ) {
      // No more blocks, skip swap in and waiting launch.
      batch_state_->step_sched_finish = true;
      KLLM_LOG_DEBUG << "No more free blocks, skip swapped and waiting queue." << *batch_state_;

      if (batch_scheduler_config_.preempt_mode == PreemptMode::SWAP) {
        KLLM_LOG_DEBUG << "running " << *req << " swapout async"
                       << ", current_block_num:" << req->kv_cache_blocks[0].size()
                       << ", current_forwarding_token_size:" << req->forwarding_tokens.size();

        // Merge all swapin request before swapout.
        if (!batch_state_->swapin_pending_requests_.empty()) {
          KLLM_LOG_DEBUG << "Pending swapin requests exists, merge it first.";
          MergePendingSwapinRequests(true, false);
        }
        size_t free_block_num = 0;
        size_t swapped_block_num = 0;
        std::vector<int> swapout_memory_blocks;
        status =
            cache_manager_->SwapoutRequestAsync(req->req_id, swapped_block_num, free_block_num, swapout_memory_blocks);
        if (status.OK()) {
          req->RebuildBlockPtrs();
          batch_state_->swapout_pending_requests_[req->req_id] = req;
          batch_state_->schedule_output->running_reqs.erase(it);

          total_needed_block_num -= step_block_num;

          // Record swapout operation.
          if (req->attn_dp_group_id == batch_state_->schedule_output->swapout_req_block_ids.size()) {
            batch_state_->schedule_output->swapout_req_block_ids.push_back(
                std::unordered_map<int64_t, std::vector<int>>());
          }
          batch_state_->schedule_output->swapout_req_block_ids[req->attn_dp_group_id][req->req_id] =
              swapout_memory_blocks;

          continue;
        }
        KLLM_LOG_ERROR << "Swap out request error, recompute it. info: " << status.GetMessage();
      }
    }

    if (!status.OK() || batch_scheduler_config_.preempt_mode == PreemptMode::RECOMPUTE) {
      KLLM_LOG_DEBUG << "running " << *req << " recompute.";

      size_t freeable_block_num = 0;
      cache_manager_->GetRequestFreeableBlockNum(req->req_id, freeable_block_num);
      // Add recomputed request to the begining of waiting queue.
      if (batch_scheduler_config_.enable_async) {
        it = AsyncRecomputeRequest(it, false);
      } else {
        it = RecomputeRequest(it, false);
      }
      total_needed_block_num -= step_block_num;
      continue;
    }

    KLLM_LOG_DEBUG << "running " << *req << " should not arrive here.";
  }

  batch_state_->MergeRunningPendingReqs(dp_max_batch_size_);
}

void ContinuousBatchingStrategy::ProcessSwappedQueue() {
  PROFILE_EVENT_SCOPE(ProcessSwappedQueue, "ProcessSwappedQueue");
  KLLM_LOG_DEBUG << "ProcessSwappedQueue invoked:" << *batch_state_
                 << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  if (batch_scheduler_config_.preempt_mode != SWAP) {
    return;
  }

  // Merge pending swapout requests.
  Status status = MergePendingSwapoutRequests(false, true);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "ProcessSwappedQueue error, info: " << status.GetMessage();
  }

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size();
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->sampling_token_num;
  }
  auto [step_token_num, step_not_kv_cached_token_num] = CheckRunningQueueStepTokens();

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

  if (batch_state_->swapped_queue.empty()) {
    scheduler_ticktok_->Skip();
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_DEBUG << "Skip swapped queue." << *batch_state_;
    scheduler_ticktok_->Skip();
    return;
  }

  for (auto it = batch_state_->swapped_queue.begin(); it != batch_state_->swapped_queue.end();) {
    auto req = it->second;

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_DEBUG << "req timeout in swapped: " << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in swapped."), true);
      it = batch_state_->swapped_queue.erase(it);

      // Record finish req_id
      if (req->attn_dp_group_id == batch_state_->schedule_output->finish_req_ids.size()) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_DEBUG << "req aborted in swapped:" << req;

      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in swapped."), true);
      it = batch_state_->swapped_queue.erase(it);

      // Record finish req_id
      if (req->attn_dp_group_id == batch_state_->schedule_output->finish_req_ids.size()) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      continue;
    }

    size_t swapin_needed_block_num = 0;
    cache_manager_->GetRequestNeededBlockNumForOneNextToken(req->req_id, swapin_needed_block_num);

    const size_t swapin_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.swapin_block_threshold);

    const size_t total_free_block_num = cache_manager_->GetUsableBlockNumber();
    const size_t step_needed_block_num =
        cache_manager_->GetRequestStepBlockNumber(req->req_id, req->forwarding_tokens.size());

    // In this context, all other db groups will be paused.
    scheduler_ticktok_->Lock();

    // The total num include other dp groups.
    step_batch_size = scheduler_shared_counter_->step_batch_size.Get();
    step_token_num = scheduler_shared_counter_->step_token_num.Get();
    step_logits_num = scheduler_shared_counter_->step_logits_num.Get();

    if (step_logits_num + req->sampling_token_num <= dp_max_logits_num_ && step_batch_size < dp_max_batch_size_ &&
        step_token_num + req->draft_tokens.size() + kStepGenerateTokenNum <= dp_max_step_token_num_ &&
        swapin_needed_block_num + step_needed_block_num + swapin_block_threshold <= total_free_block_num) {
      // Assume the operation will be successful, fallback if failed, so that SwapinRequestAsync is not blocked.
      scheduler_shared_counter_->step_batch_size.Increase(1);
      scheduler_shared_counter_->step_token_num.Increase(req->draft_tokens.size() + kStepGenerateTokenNum);
      scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
      scheduler_ticktok_->Unlock();

      // Merge pending swapout requests before swap in.
      if (!batch_state_->swapout_pending_requests_.empty()) {
        KLLM_LOG_DEBUG << "Pending swapout requests exists, merge it first.";
        MergePendingSwapoutRequests(true, false);
      }
      std::vector<int> swapin_memory_blocks;
      status = cache_manager_->SwapinRequestAsync(req->req_id, swapin_needed_block_num, req->kv_cache_blocks,
                                                  swapin_memory_blocks);
      if (status.OK()) {
        batch_state_->swapin_pending_requests_[req->req_id] = req;
        it = batch_state_->swapped_queue.erase(it);

        // Record swapin operation.
        if (req->attn_dp_group_id == batch_state_->schedule_output->swapin_req_block_ids.size()) {
          batch_state_->schedule_output->swapin_req_block_ids.push_back(
              std::unordered_map<int64_t, std::vector<int>>());
        }
        batch_state_->schedule_output->swapin_req_block_ids[req->attn_dp_group_id][req->req_id] = swapin_memory_blocks;

        continue;
      } else {
        // decrease counter if failed, thread-safe, no timing-control needed.
        scheduler_shared_counter_->step_batch_size.Decrease(1);
        scheduler_shared_counter_->step_token_num.Decrease(req->draft_tokens.size() + kStepGenerateTokenNum);
        scheduler_shared_counter_->step_logits_num.Decrease(req->sampling_token_num);
      }

      KLLM_LOG_ERROR << "Swap in request error, info: " << status.GetMessage();
      ++it;
    } else {
      scheduler_ticktok_->Unlock();
    }

    // Swapped job still existed, skip launch waiting.
    batch_state_->step_sched_finish = true;
    KLLM_LOG_DEBUG << "Swapped queue not empty, skip processing waiting_queue." << *batch_state_;
    break;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();
}

/**
 * Processes a request to determine the appropriate number of tokens to split or fuse based on the current
 * batching strategy configuration. This function adjusts the number of output tokens in the request to match
 * the calculated split or fuse token count, and updates the shared and unique block counts accordingly.
 *
 * The function aims to optimize the processing of requests by dynamically adjusting the number of tokens
 * to be processed together, based on the configured thresholds and the current state of the request and
 * batch scheduler.
 */
bool ContinuousBatchingStrategy::ProcessSplitFuseToken(std::shared_ptr<InferRequest> req, size_t &shared_block_num,
                                                       size_t &unique_block_num, size_t &shared_token_num,
                                                       const size_t step_not_kv_cached_token_num,
                                                       const size_t decode_request_num) {
  const size_t kSplitFuseTokenNum = batch_scheduler_config_.split_fuse_token_num;
  const size_t not_kv_cached_token_num = req->forwarding_tokens.size() - shared_token_num;
  const size_t split_fuse_remain_token_num = kSplitFuseTokenNum - step_not_kv_cached_token_num;  // allow overflow
  // Skip under these conditions:
  // 1. Split fuse is disable (kSplitFuseTokenNum == 0)
  // 2. Current step already meets quota (step_not_kv_cached_token_num >= kSplitFuseTokenNum)
  // 3. No decode requests (running queue empty), add complete request directly
  // 4. Enough space for the complete request (not_kv_cached_token_num <= split_fuse_remain_token_num)
  if (step_not_kv_cached_token_num >= kSplitFuseTokenNum || decode_request_num == 0 ||
      not_kv_cached_token_num <= split_fuse_remain_token_num) {
    return false;
  }
  // If remain space less than one block, skip processing this request and halt further scheduling.
  if (split_fuse_remain_token_num < req->block_token_num) {
    return true;
  }

  // token num is align to block_token_num. (split_fuse_remain_token_num < not_kv_cached_token_num now)
  const size_t tokens_to_split = split_fuse_remain_token_num / req->block_token_num * req->block_token_num;
  KLLM_LOG_DEBUG << *req << " shared_block_num " << shared_block_num << " unique_block_num " << unique_block_num
                 << " shared_token_num " << shared_token_num << " current_token_num " << req->forwarding_tokens.size()
                 << " tokens_to_split " << tokens_to_split;

  // split forwarding_tokens
  req->forwarding_tokens.resize(shared_token_num + tokens_to_split);
  cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->forwarding_tokens, shared_block_num, unique_block_num,
                                              shared_token_num);
  // The `unique_block_num` is decremented here because, during split_fuse operations,
  // there is no need to pre-allocate a block for decoding purposes.
  --unique_block_num;
  return false;
}

void ContinuousBatchingStrategy::ProcessWaitingQueue() {
  PROFILE_EVENT_SCOPE(ProcessWaitingQueue, "ProcessWaitingQueue");
  KLLM_LOG_DEBUG << "ProcessWaitingQueue invoked, waiting queue size:" << batch_state_->waiting_queue.size()
                 << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  const size_t decode_request_num = batch_state_->schedule_output->running_reqs.size();
  auto [step_token_num, step_not_kv_cached_token_num] = CheckRunningQueueStepTokens();

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size() + batch_state_->transfer_queue.size();
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->sampling_token_num;
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
    KLLM_LOG_DEBUG << "Skip processing waiting_queue." << *batch_state_;
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
    if (req->forwarding_tokens.empty()) {  // new request
      req->forwarding_tokens = req->output_tokens;
    }

    // When the logits_custom_length is greater than 0, the size of logits to be calculated is logits_custom_length.
    if (req->logits_custom_length > 0) {
      req->sampling_token_num = req->logits_custom_length;
    }

    // Check timeout, no finished req in waiting queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_DEBUG << "req timeout in waiting:" << req;

      StopRequest(req, Status(RET_REQUEST_TIMEOUT, "timeout in waiting."), false);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_DEBUG << "req aborted in waiting:" << req;
      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in waiting."), false);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    size_t shared_block_num = 0;
    size_t unique_block_num = 0;
    size_t shared_token_num = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->forwarding_tokens, shared_block_num, unique_block_num,
                                                shared_token_num);
    if (ProcessSplitFuseToken(req, shared_block_num, unique_block_num, shared_token_num, step_not_kv_cached_token_num,
                              decode_request_num)) {
      break;
    }

    req->prefix_cache_len = shared_token_num;
    req->is_use_prefix_cache = shared_token_num > 0;

    if (req->is_use_prefix_cache) {
      REPORT_COUNTER("prefix_cache_hit_req_num", static_cast<size_t>(1));
      REPORT_COUNTER("prefix_cache_hit_token_num", shared_token_num);
      REPORT_COUNTER("prefix_cache_hit_block_num", shared_block_num);
    }

    const size_t current_token_num = req->forwarding_tokens.size();
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
        step_token_num + current_token_num <= dp_max_step_token_num_ &&
        unique_block_num + launch_block_threshold <= total_free_block_num) {
      // Assume we cound succ, so that the AllocateRequestBlocks is not blocked.
      scheduler_shared_counter_->step_batch_size.Increase(1);
      scheduler_shared_counter_->step_token_num.Increase(current_token_num);
      scheduler_shared_counter_->step_logits_num.Increase(req->sampling_token_num);
      scheduler_ticktok_->Unlock();

      Status status = cache_manager_->AllocateRequestBlocks(req->req_id, unique_block_num, req->kv_cache_blocks);
      if (status.OK()) {
        step_not_kv_cached_token_num += current_token_num - shared_token_num;
        req->RebuildBlockPtrs();

        // if full matched, skip decode and put it to the end of decode list.
        if (shared_token_num == req->forwarding_tokens.size()) {
          KLLM_LOG_DEBUG << "Full matched, skip prefill, " << *req;
          REPORT_COUNTER("full_prompt_matched_req_num", static_cast<size_t>(1));
          REPORT_COUNTER("full_prompt_matched_block_num", shared_block_num);

          req->infer_stage = InferStage::STATE_DECODE;
          req->kv_cached_token_num = shared_token_num - kStepGenerateTokenNum;
          req->prefix_cache_len = req->kv_cached_token_num;
          req->is_use_prefix_cache = false;
          batch_state_->schedule_output->running_reqs.insert(
              batch_state_->schedule_output->running_reqs.begin() + decode_request_num, req);
        } else {
          KLLM_LOG_DEBUG << "shared token not equal forwaing size, " << *req;
          req->kv_cached_token_num = shared_token_num;
          if (connector_config_.group_role == GroupRole::DECODE) {
            batch_state_->transfer_queue.emplace_back(req);
            it = batch_state_->waiting_queue.erase(it);
            KLLM_LOG_DEBUG << "Decode put req to transfer queue, req id: " << req->kv_comm_request_id;
            continue;
          } else {
            batch_state_->schedule_output->running_reqs.emplace_back(req);
            KLLM_LOG_DEBUG << "Prefill put req to running queue, req id: " << req->kv_comm_request_id;
            DetermineDraftNum(req);
          }
        }

        it = batch_state_->waiting_queue.erase(it);
        if (batch_scheduler_config_.split_fuse_token_num > 0 &&
            step_not_kv_cached_token_num >= batch_scheduler_config_.split_fuse_token_num) {
          break;
        }
        // The flexible cache handling could be placed prior to the split_fuse break. Try moving it after testing.
        cache_manager_->UpdateFlexibleCache(req->req_id, req->forwarding_tokens, shared_token_num,
                                            req->flexible_cached_copy_tasks);
        continue;
      } else {
        // decrease counter if failed, thread-safe, no timing-control needed.
        scheduler_shared_counter_->step_batch_size.Decrease(1);
        scheduler_shared_counter_->step_token_num.Decrease(current_token_num);
        scheduler_shared_counter_->step_logits_num.Decrease(req->sampling_token_num);

        KLLM_LOG_ERROR << "Alllocate blocks error, waiting req can not be launched, " << *req << status.GetMessage();
      }
      KLLM_LOG_DEBUG << "Waiting all pending swapout requests done, and stay in waiting.";
      MergePendingSwapoutRequests(true, false);
    } else {
      scheduler_ticktok_->Unlock();
    }

    KLLM_LOG_DEBUG << "total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                   << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
    break;
  }

  // Current dp group finished, remove from loop list.
  scheduler_ticktok_->Skip();
}

Status ContinuousBatchingStrategy::MergePendingSwapinRequests(bool blocking, bool early_stop) {
  size_t swapin_left_req_num = 0;
  do {
    std::vector<int64_t> swapin_req_ids;
    Status status = cache_manager_->WaitSwapinRequests(swapin_req_ids, swapin_left_req_num, blocking);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Error MergePendingSwapinRequests WaitSwapinRequests failed. swapin_req_ids:" << swapin_req_ids
                     << ", info: " << status.GetMessage();
      return status;
    }
    if (!swapin_req_ids.empty()) {
      KLLM_LOG_DEBUG << "finished swapin request size:" << swapin_req_ids.size();
    }
    for (int64_t req_id : swapin_req_ids) {
      auto it = batch_state_->swapin_pending_requests_.find(req_id);
      if (it == batch_state_->swapin_pending_requests_.end()) {
        KLLM_LOG_ERROR << "The cached swapin req_id:" << req_id << " is not found in pending queue.";
        continue;
      }

      auto &req = it->second;
      status = cache_manager_->MergeSwapinRequest(req->req_id, req->kv_cache_blocks);
      if (!status.OK()) {
        KLLM_LOG_DEBUG << "Error MergeSwapinRequest " << *req << ", info: " << status.GetMessage();
        return status;
      }

      // Record merged swapin request.
      batch_state_->merged_swapin_req_ids.push_back(req->req_id);

      KLLM_LOG_DEBUG << "after finish swapin req " << req->req_id
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_forwarding_token_num:" << req->forwarding_tokens.size();
      batch_state_->running_pending_reqs.push_back(req);
      batch_state_->swapin_pending_requests_.erase(it);
    }
  } while (!early_stop && swapin_left_req_num > 0);

  if (!batch_state_->merged_swapin_req_ids.empty()) {
    KLLM_LOG_DEBUG << "After merge merged_swapin_req_ids. size=" << batch_state_->merged_swapin_req_ids.size()
                   << ", ids=" << Vector2Str(batch_state_->merged_swapin_req_ids);
  }

  return Status();
}

Status ContinuousBatchingStrategy::MergePendingSwapoutRequests(bool blocking, bool early_stop) {
  // Wait all requests done.
  size_t swapout_left_req_num = 0;
  do {
    if (blocking) {
      KLLM_LOG_DEBUG << "multi_batch_id=" << batch_state_->multi_batch_id_
                     << "before WaitSwapoutRequests with blocking=true, swapout_pending_requests_.size="
                     << batch_state_->swapout_pending_requests_.size();
    }
    std::vector<int64_t> swapout_req_ids;
    Status status = cache_manager_->WaitSwapoutRequests(swapout_req_ids, swapout_left_req_num, blocking);
    if (!status.OK()) {
      KLLM_LOG_DEBUG << "multi_batch_id=" << batch_state_->multi_batch_id_
                     << "Error MergePendingSwapoutRequests WaitSwapoutRequests failed. swapout_req_ids:"
                     << swapout_req_ids << ", info: " << status.GetMessage();
      return status;
    }
    if ((!swapout_req_ids.empty()) || blocking) {
      KLLM_LOG_DEBUG << "multi_batch_id=" << batch_state_->multi_batch_id_
                     << "finished swapout request size:" << swapout_req_ids.size();
    }
    for (int64_t req_id : swapout_req_ids) {
      auto it = batch_state_->swapout_pending_requests_.find(req_id);
      if (it == batch_state_->swapout_pending_requests_.end()) {
        KLLM_LOG_ERROR << "The cached swapout req_id:" << req_id << " is not found in pending queue.";
        continue;
      }

      auto &req = it->second;
      status = cache_manager_->MergeSwapoutRequest(req->req_id);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "The cached swapout :" << *req << " failed.";
        return status;
      }

      // Record merged swapout request.
      batch_state_->merged_swapout_req_ids.push_back(req->req_id);

      KLLM_LOG_DEBUG << "multi_batch_id=" << batch_state_->multi_batch_id_ << "after finish swapout req " << req->req_id
                     << ", current_block_num:" << req->kv_cache_blocks[0].size() << batch_state_;

      batch_state_->swapped_queue[req->req_id] = req;
      batch_state_->swapout_pending_requests_.erase(it);
    }
  } while (!early_stop && swapout_left_req_num > 0);

  if (!batch_state_->merged_swapout_req_ids.empty()) {
    KLLM_LOG_DEBUG << "After merge merged_swapout_req_ids. size=" << batch_state_->merged_swapout_req_ids.size()
                   << ", ids=" << Vector2Str(batch_state_->merged_swapout_req_ids);
  }

  return Status();
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

void ContinuousBatchingStrategy::UpdateSwapPendingRequests() {
  bool blocking = false;
  bool early_stop = false;
  MergePendingSwapinRequests(blocking, early_stop);
  MergePendingSwapoutRequests(blocking, early_stop);
}

void ContinuousBatchingStrategy::Schedule(std::vector<std::shared_ptr<InferRequest>> &waiting_reqs) {
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
  ProcessRunningQueue();
  REPORT_METRIC("batch_scheduler_running_queue_time_us", ProfileTimer::GetCurrentTimeInUs() - start_us);
  ProcessSwappedQueue();
  REPORT_METRIC("batch_scheduler_swapped_queue_time_us", ProfileTimer::GetCurrentTimeInUs() - start_us);
  ProcessWaitingQueue();
  REPORT_METRIC("batch_scheduler_waiting_queue_time_us", ProfileTimer::GetCurrentTimeInUs() - start_us);
  ProcessTransferQueue();
  REPORT_METRIC("batch_scheduler_transfer_queue_time_us", ProfileTimer::GetCurrentTimeInUs() - start_us);

  // Must barrier before reorder.
  scheduler_ticktok_->Barrier();

  // Change next visit order of dp groups, for load balance.
  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reorder();
  }

  REPORT_COUNTER("batch_scheduler_pending_swapin_size", batch_state_->swapin_pending_requests_.size());
  REPORT_COUNTER("batch_scheduler_pending_swapout_size", batch_state_->swapout_pending_requests_.size());
  if (batch_state_->schedule_output->running_reqs.size() > 0) {
    // This output will be executed. send swap waiting info to workers
    batch_state_->schedule_output->merged_swapin_req_ids.resize(1);
    batch_state_->schedule_output->merged_swapin_req_ids[0] = batch_state_->merged_swapin_req_ids;
    KLLM_LOG_DEBUG << "Add merged_swapin_req_ids size=" << batch_state_->merged_swapin_req_ids.size()
                   << ", ids=" << Vector2Str(batch_state_->merged_swapin_req_ids);
    batch_state_->merged_swapin_req_ids.clear();
    batch_state_->schedule_output->merged_swapout_req_ids.resize(1);
    batch_state_->schedule_output->merged_swapout_req_ids[0] = batch_state_->merged_swapout_req_ids;
    KLLM_LOG_DEBUG << "Add merged_swapout_req_ids size=" << batch_state_->merged_swapout_req_ids.size()
                   << ", ids=" << Vector2Str(batch_state_->merged_swapout_req_ids);
    batch_state_->merged_swapout_req_ids.clear();
  }
}

}  // namespace ksana_llm
