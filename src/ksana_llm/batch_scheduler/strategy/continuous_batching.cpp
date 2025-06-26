/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
#include <cmath>
#include <memory>

#include "base_strategy.h"
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

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num)
    : BaseScheduleStrategy(batch_scheduler_config, tp_num) {
  auto env = Singleton<Environment>::GetInstance();

  env->GetConnectorConfigs(connector_config_);
  if (connector_config_.group_role != GroupRole::NONE) {
    TransferEngine::GetInstance()->Initialize(connector_config_.group_role);
  }

  if (env->IsFlashMlaEnable() && IsAbsorbWeightsEnabled()) {
    decode_token_num_threshold_ = 2;  // input_ids <= 2 will regard as decode, using page attention
  }

  dp_max_step_token_num_ = batch_scheduler_config_.max_step_token_num / env->GetAttnDataParallelSize();
  dp_max_batch_size_ = batch_scheduler_config_.max_batch_size / env->GetAttnDataParallelSize();
  dp_max_logits_num_ = dp_max_batch_size_ * batch_scheduler_config.max_decode_tokens_per_req;
  if (connector_config_.group_role == GroupRole::DECODE) {
    dp_max_decode_batch_size_ = dp_max_batch_size_;
    // 增加预参数的大小
    dp_max_batch_size_ = (batch_scheduler_config_.max_batch_size + batch_scheduler_config_.max_pretransfer_batch_size) /
                         env->GetAttnDataParallelSize();
    // Decode 无需限制 dp_max_step_token_num_
    dp_max_step_token_num_ *= dp_max_batch_size_;
    KLLM_LOG_INFO << "decode dp_max_batch_size_:" << dp_max_batch_size_
                  << ", dp_max_decode_batch_size_:" << dp_max_decode_batch_size_;
  }
}

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >= req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms;
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
      req->output_tokens.size() >= batch_scheduler_config_.max_token_len ||
      (req->req_fsm != nullptr && req->req_fsm->IsStopState(req->fsm_state_id))) {
    stop_checker_->CheckCompleteStopStrings(req);
    return true;
  }

  // When stop strings are checked and matched, stop early
  return stop_checker_->CheckIncrementalStopStrings(req);
}

void ContinuousBatchingStrategy::ExtendTokensWithRetokenization(std::shared_ptr<InferRequest> req) {
  // First, convert the output_tokens into a string A (excluding the input_tokens part).
  // Then, concatenate the original prompt, the string A, and the fixed string defined in the structured output regex.
  // Finally, the combined string is encoded using a tokenizer to obtain the new output_tokens.
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);

  // detokenize the newly generated token sequence to obtain the corresponding string.
  std::string structure_text = "";
  std::vector<int> new_tokens;
  if (req->output_tokens.size() > req->input_tokens.size()) {
    new_tokens = std::vector<int>(req->output_tokens.begin() + req->input_tokens.size(), req->output_tokens.end());
    Singleton<Tokenizer>::GetInstance()->Decode(new_tokens, structure_text);
  }

  // collapse consecutive non-generation states into one.
  while (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    structure_text += state->GetEdge().first;
    state = req->req_fsm->GetState(state->GetNextStateId());
  }

  // Clear the request for assignment.
  req->output_tokens = req->input_tokens;
  new_tokens.clear();

  // Assign values.
  Singleton<Tokenizer>::GetInstance()->Encode(structure_text, new_tokens, /* add_special_tokens */ false);
  // TODO(robertyuan): if tokens have kv-caches are replaced, kv_cached_token_num should be adjusted.
  req->output_tokens.insert(req->output_tokens.end(), new_tokens.begin(), new_tokens.end());

  // add new tokens to forwarding_tokens, their kv-cache will be generated.
  req->forwarding_tokens = req->output_tokens;

  req->NotifyStep();

  // State transition.
  req->fsm_state_id = state->state_id_;
}

void ContinuousBatchingStrategy::ExtendTokensWithoutRetokenization(std::shared_ptr<InferRequest> req) {
  // Unlike the ExtendTokensWithRetokenization, in this function, the fixed string defined in the structured
  // output regex has already been preconverted into the corresponding token_id_list. Therefore, we only need to
  // append the new token to the output_tokens.
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);

  while (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    std::vector<int> jump_tokens = req->req_fsm->GetStringTokens(state->GetEdge().first);
    req->output_tokens.insert(req->output_tokens.end(), jump_tokens.begin(), jump_tokens.end());
    state = req->req_fsm->GetState(state->GetNextStateId());
  }

  // add new tokens to forwarding_tokens, their kv-cache will be generated.
  req->forwarding_tokens = req->output_tokens;

  req->NotifyStep();
  req->fsm_state_id = state->state_id_;
}

void ContinuousBatchingStrategy::JumpForwardRequest(std::shared_ptr<InferRequest> req) {
  // When the request is in a Non-Generatrion state, perform a Jump-Forward using a constant string.
  if (req->req_fsm.get() == nullptr) {
    return;
  }
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);
  if (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    // Refer to https://lmsys.org/blog/2024-02-05-compressed-fsm/#tokenization-boundary-handling
    // We use retokenizer to avoid issues related to tokenization boundaries during constrained decoding.
    ExtendTokensWithRetokenization(req);
  }
}

void ContinuousBatchingStrategy::ProcessStructuredOutput(std::shared_ptr<InferRequest> req) {
  // Determine that whether the request can transition to the next state and perform a Jump-Forward
  if (req->req_fsm.get() == nullptr) {
    return;
  }
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);
  if (state->state_type_ == FiniteStateType::GENERATION_STATE) {
    size_t next_state_id = state->GetNextStateId(req->output_tokens.back());
    if (next_state_id != req->fsm_state_id) {
      // The requested end token satisfies the jump condition, and the request will transition to the next state.
      if (Singleton<Environment>::GetInstance()->IsPrefixCachingEnabled()) {
        req->prefix_cache_len = req->output_tokens.size() - 1;
      }
      req->req_fsm->CheckFSMPopToken(req->fsm_state_id, req->output_tokens);
      req->fsm_state_id = next_state_id;
      JumpForwardRequest(req);
    }
  }
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
  KLLM_LOG_DEBUG << "RecomputeRequest " << req;

  // Add request to the begining of waiting queue.
  req->kv_cache_blocks.clear();
  req->kv_cache_blocks.resize(Singleton<Environment>::GetInstance()->GetAttentionTensorParallel());
  req->infer_stage = InferStage::STAGE_CONTEXT;
  req->step = 0;
  req->kv_cached_token_num = 0;
  req->suggested_draft_num = 0;
  req->prefix_cache_len = 0;

  static constexpr bool terminate = false;
  ResetRequest(req, Status(RET_SUCCESS, "RecomputeRequest"), is_swap_req, terminate);

  batch_state_->waiting_queue.emplace_front(req);
  return batch_state_->schedule_output->running_reqs.erase(it);
}

void ContinuousBatchingStrategy::StopRequest(std::shared_ptr<InferRequest> req, Status req_status, bool is_swap_req) {
  ResetRequest(req, req_status, is_swap_req, true);
}

void ContinuousBatchingStrategy::ResetRequest(std::shared_ptr<InferRequest> req, Status req_status, bool is_swap_req,
                                              bool terminate) {
  KLLM_LOG_DEBUG << "ResetRequest " << *req << ", req_status:" << req_status.ToString()
                 << ", is_swap_req:" << is_swap_req << ", terminate:" << terminate;

  req->finish_status = req_status;
  req->finished = terminate;

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
  size_t step_token_num = 0, step_not_kv_cached_token_num = 0;
  size_t total_sampling_token_num = 0, req_num = 0;
  for (auto it = batch_state_->schedule_output->running_reqs.begin();
       it != batch_state_->schedule_output->running_reqs.end();) {
    const auto &req = *it;
    const size_t not_kv_cached_token_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
    const size_t req_token_num = not_kv_cached_token_num <= decode_token_num_threshold_ ? not_kv_cached_token_num
                                                                                        : req->forwarding_tokens.size();
    if (step_token_num + req_token_num > dp_max_step_token_num_ ||
        total_sampling_token_num + req->sampling_token_num > dp_max_logits_num_ || req_num >= dp_max_batch_size_) {
      it = RecomputeRequest(it);
      continue;
    }
    step_token_num += req_token_num;
    step_not_kv_cached_token_num += not_kv_cached_token_num;
    total_sampling_token_num += req->sampling_token_num;
    ++req_num;
    ++it;
  }
  return {step_token_num, step_not_kv_cached_token_num};
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
    req->forwarding_tokens.resize(req->forwarding_tokens.size() - req->forwarding_tokens_draft_num +
                                  req->accepted_tokens.size());
    // current token has kv_cache
    req->kv_cached_token_num = req->forwarding_tokens.size();
    req->prefix_cache_len = req->kv_cached_token_num;

    // Always update cache manager, even if request is finished.
    Status status = cache_manager_->UpdateRequestTokens(req->req_id, req->forwarding_tokens, req->kv_cached_token_num,
                                                        req->kv_cache_blocks);

    if (req->forwarding_tokens.size() - req->accepted_tokens.size() < req->output_tokens.size()) {
      // When the request actually needs to calculate Multi-Token, insert the request into the waiting queue.
      // This is primarily used for the split-fuse feature.
      req->infer_stage = InferStage::STAGE_CONTEXT;
      req->forwarding_tokens = req->output_tokens;
      req->draft_tokens.clear();
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      req->sampling_token_num = kStepGenerateTokenNum;
      batch_state_->waiting_queue.emplace_front(req);
      it = batch_state_->schedule_output->running_reqs.erase(it);
      KLLM_LOG_DEBUG << "splitfuse cal multi-token" << req;
      continue;
    }

    // append generated token
    req->forwarding_tokens.emplace_back(req->generated_token);
    req->sampling_token_num = kStepGenerateTokenNum;

    // append new tokens to output_tokens
    req->output_mutex.lock();
    req->output_tokens.insert(req->output_tokens.end(),
                              req->forwarding_tokens.end() - req->accepted_tokens.size() - kStepGenerateTokenNum,
                              req->forwarding_tokens.end());
    req->output_mutex.unlock();

    req->req_ctx->emplace("status_code", std::to_string(static_cast<int>(req->finish_status.GetCode())));
    std::unordered_map<std::string, std::string> filtered_ctx = *req->req_ctx;
    filtered_ctx.erase("kv-comm-request-id");
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(filtered_ctx);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "UpdateRequestTokens " << req << " error, recompute it, info: " << status.GetMessage();
      req->draft_tokens.clear();
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      it = RecomputeRequest(it);
      continue;
    }

    // Checking if the request can perform a StructuredOutput optimization.
    ProcessStructuredOutput(req);

    // Check if finished.
    if (CheckRequestFinish(req)) {
      const auto end_time = ProfileTimer::GetCurrentTimeInMs();
      const size_t output_token_num = req->output_tokens.size();
      const uint64_t duration = end_time - req->timestamp_in_ms;
      if (req->finish_status.GetCode() == RET_SUCCESS) {
        REPORT_METRIC(forward_cost_time_ms, duration, attributes);
        REPORT_METRIC(metric_output_token_num, output_token_num, attributes);
        if (output_token_num == 0) {
          REPORT_COUNTER(metric_zero_output_token_num, static_cast<size_t>(1), attributes);
        }
      } else {
        REPORT_COUNTER(forward_req_error_num, static_cast<size_t>(1), attributes);
      }

      // TODO(shawnding): Adjust to microsecond precision
      if (duration != 0) {
        REPORT_METRIC(time_to_per_output_token_ms, output_token_num / duration, attributes);
      } else {
        KLLM_LOG_DEBUG << fmt::format(
            "Req duration is zero, req_id: {}, input_token_num: {}, output_token_num: {}, "
            "req start time is: {}, req end time is: {}",
            req->req_id, req->input_tokens.size(), output_token_num, req->timestamp_in_ms, end_time);
        REPORT_METRIC(time_to_per_output_token_ms, output_token_num, attributes);
      }

      // Record finish req_id
      if (req->attn_dp_group_id >= batch_state_->schedule_output->finish_req_ids.size()) {
        size_t needed_push_size = req->attn_dp_group_id - batch_state_->schedule_output->finish_req_ids.size() + 1;
        for (size_t idx = 0; idx < needed_push_size; ++idx) {
          batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{});
        }
      }
      batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);

      StopRequest(req, Status(RET_SUCCESS), false);
      if (connector_config_.group_role == GroupRole::PREFILL) {
        batch_state_->transfer_queue.emplace_back(req);
      }
      it = batch_state_->schedule_output->running_reqs.erase(it);

      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      REPORT_COUNTER(forward_req_timeout_num, static_cast<size_t>(1), attributes);
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
      KLLM_LOG_WARNING << "req aborted in running: " << req;

      StopRequest(req, Status(RET_REQUEST_TERMINATED, "req aborted in running."), false);
      it = batch_state_->schedule_output->running_reqs.erase(it);

      // Record finish req_id
      if (req->attn_dp_group_id == batch_state_->schedule_output->finish_req_ids.size()) {
        batch_state_->schedule_output->finish_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->finish_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      REPORT_COUNTER(forward_req_aborted_num, static_cast<size_t>(1), attributes);

      continue;
    }

    // Not finished, notify streaming iterator.
    req->NotifyStep();

    // append draft tokens
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
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
  for (auto &req : batch_state_->schedule_output->running_reqs) {
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
    size_t swapout_block_threshold = std::ceil(running_batch_size * batch_scheduler_config_.swapout_block_threshold);

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
    if (batch_state_->waiting_queue.size() > 0 && batch_scheduler_config_.split_fuse_token_num > 0) {
      for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end(); it++) {
        auto req = *it;
        if (req->kv_cache_blocks[0].size() > 0) {
          // Note(TJ): maybe cloud use StopRequest
          cache_manager_->DestroyFinishedRequest(req->req_id);
          req->kv_cache_blocks.clear();
          req->kv_cache_blocks.resize(Singleton<Environment>::GetInstance()->GetAttentionTensorParallel());
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
      RecomputeRequest(it);
      total_needed_block_num -= step_block_num;
      continue;
    }

    KLLM_LOG_DEBUG << "running " << *req << " should not arrive here.";
  }

  batch_state_->MergeRunningPendingReqs();
}

void ContinuousBatchingStrategy::ProcessSwappedQueue() {
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

  if (batch_state_->swapped_queue.empty()) {
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_DEBUG << "Skip swapped queue." << *batch_state_;
    return;
  }

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size();
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->sampling_token_num;
  }
  auto [step_token_num, step_not_kv_cached_token_num] = CheckRunningQueueStepTokens();
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

    if (step_logits_num + req->sampling_token_num <= dp_max_logits_num_ && step_batch_size < dp_max_batch_size_ &&
        step_token_num + req->draft_tokens.size() + kStepGenerateTokenNum <= dp_max_step_token_num_ &&
        swapin_needed_block_num + step_needed_block_num + swapin_block_threshold <= total_free_block_num) {
      // Merge pending swapout requests before swap in.
      if (!batch_state_->swapout_pending_requests_.empty()) {
        KLLM_LOG_DEBUG << "Pending swapout requests exists, merge it first.";
        MergePendingSwapoutRequests(true, false);
      }
      std::vector<int> swapin_memory_blocks;
      status = cache_manager_->SwapinRequestAsync(req->req_id, swapin_needed_block_num, req->kv_cache_blocks,
                                                  swapin_memory_blocks);
      if (status.OK()) {
        ++step_batch_size;
        step_logits_num += req->sampling_token_num;
        step_token_num += req->draft_tokens.size() + kStepGenerateTokenNum;

        batch_state_->swapin_pending_requests_[req->req_id] = req;
        it = batch_state_->swapped_queue.erase(it);

        // Record swapin operation.
        if (req->attn_dp_group_id == batch_state_->schedule_output->swapin_req_block_ids.size()) {
          batch_state_->schedule_output->swapin_req_block_ids.push_back(
              std::unordered_map<int64_t, std::vector<int>>());
        }
        batch_state_->schedule_output->swapin_req_block_ids[req->attn_dp_group_id][req->req_id] = swapin_memory_blocks;

        continue;
      }
      KLLM_LOG_ERROR << "Swap in request error, info: " << status.GetMessage();
      ++it;
    }

    // Swapped job still existed, skip launch waiting.
    batch_state_->step_sched_finish = true;
    KLLM_LOG_DEBUG << "Swapped queue not empty, skip processing waiting_queue." << *batch_state_;
    break;
  }
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
  KLLM_LOG_DEBUG << "ProcessWaitingQueue invoked, waiting queue size:" << batch_state_->waiting_queue.size()
                 << ", total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();

  if (batch_state_->waiting_queue.empty()) {
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_DEBUG << "Skip processing waiting_queue." << *batch_state_;
    return;
  }

  const size_t decode_request_num = batch_state_->schedule_output->running_reqs.size();
  auto [step_token_num, step_not_kv_cached_token_num] = CheckRunningQueueStepTokens();

  size_t step_batch_size = batch_state_->schedule_output->running_reqs.size();
  size_t step_logits_num = 0;
  for (const auto &req : batch_state_->schedule_output->running_reqs) {
    step_logits_num += req->sampling_token_num;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    req->cache_manager = cache_manager_;
    JumpForwardRequest(req);

    if (req->req_fsm != nullptr && CheckRequestFinish(req)) {
      KLLM_LOG_DEBUG << "stop req_id:" << req->req_id << " finished.";
      StopRequest(req, Status(RET_SUCCESS), false);
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }
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
      REPORT_COUNTER(prefix_cache_hit_req_num, static_cast<size_t>(1));
      REPORT_COUNTER(prefix_cache_hit_token_num, shared_token_num);
      REPORT_COUNTER(prefix_cache_hit_block_num, shared_block_num);
    }

    const size_t current_token_num = req->forwarding_tokens.size();
    const size_t launch_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.launch_block_threshold);

    // Get usable block number every time, the blocks matched by req are not reusable here.
    const size_t total_free_block_num = cache_manager_->GetRequestUsableBlockNumber(req->req_id);
    if (step_logits_num + req->sampling_token_num <= dp_max_logits_num_ && step_batch_size < dp_max_batch_size_ &&
        step_token_num + current_token_num <= dp_max_step_token_num_ &&
        unique_block_num + launch_block_threshold <= total_free_block_num) {
      Status status = cache_manager_->AllocateRequestBlocks(req->req_id, unique_block_num, req->kv_cache_blocks);
      if (status.OK()) {
        ++step_batch_size;
        step_logits_num += req->sampling_token_num;
        step_token_num += current_token_num;
        step_not_kv_cached_token_num += current_token_num - shared_token_num;

        // if full matched, skip decode and put it to the end of decode list.
        if (shared_token_num == req->forwarding_tokens.size()) {
          KLLM_LOG_DEBUG << "Full matched, skip prefill, " << *req;
          REPORT_COUNTER(full_prompt_matched_req_num, static_cast<size_t>(1));
          REPORT_COUNTER(full_prompt_matched_block_num, shared_block_num);
          REPORT_METRIC(time_to_first_token_ms, ProfileTimer::GetCurrentTimeInMs() - req->timestamp_in_ms);

          req->infer_stage = InferStage::STATE_DECODE;
          req->kv_cached_token_num = shared_token_num - kStepGenerateTokenNum;
          req->prefix_cache_len = req->kv_cached_token_num;
          req->is_use_prefix_cache = false;
          batch_state_->schedule_output->running_reqs.insert(
              batch_state_->schedule_output->running_reqs.begin() + decode_request_num, req);
        } else {
          KLLM_LOG_DEBUG << "shared token not equal forwaing size, " << req;
          req->kv_cached_token_num = shared_token_num;
          req->mtp_kv_cached_token_num = req->kv_cached_token_num;
          if (connector_config_.group_role == GroupRole::DECODE) {
            batch_state_->transfer_queue.emplace_back(req);
            it = batch_state_->waiting_queue.erase(it);
            continue;
          } else {
            batch_state_->schedule_output->running_reqs.emplace_back(req);
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
        KLLM_LOG_ERROR << "Alllocate blocks error, waiting req can not be launched, " << *req << status.GetMessage();
      }
      KLLM_LOG_DEBUG << "Waiting all pending swapout requests done, and stay in waiting.";
      MergePendingSwapoutRequests(true, false);
    }
    KLLM_LOG_DEBUG << "total_free_block_num:" << cache_manager_->GetUsableBlockNumber()
                   << ", future_free_block_num:" << cache_manager_->GetFutureFreeBlockNumber();
    break;
  }
}

Status ContinuousBatchingStrategy::MergePendingSwapinRequests(bool blocking, bool early_stop) {
  KLLM_LOG_DEBUG << "MergePendingSwapinRequests invoked. " << *batch_state_ << ", blocking:" << blocking
                 << ", early_stop:" << early_stop;
  // Wait all requests done.
  size_t swapin_left_req_num = 0;
  do {
    std::vector<int64_t> swapin_req_ids;
    Status status = cache_manager_->WaitSwapinRequests(swapin_req_ids, swapin_left_req_num, blocking);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Error MergePendingSwapinRequests WaitSwapinRequests failed. swapin_req_ids:" << swapin_req_ids
                     << ", info: " << status.GetMessage();
      return status;
    }

    KLLM_LOG_DEBUG << "finished swapin request size:" << swapin_req_ids.size();
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
      if (req->attn_dp_group_id == batch_state_->schedule_output->merged_swapin_req_ids.size()) {
        batch_state_->schedule_output->merged_swapin_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->merged_swapin_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      KLLM_LOG_DEBUG << "MergePendingSwapinRequests swap in " << req
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_forwarding_token_num:" << req->forwarding_tokens.size();
      batch_state_->running_pending_reqs.push_back(req);
      batch_state_->swapin_pending_requests_.erase(it);
    }
  } while (!early_stop && swapin_left_req_num > 0);

  return Status();
}

Status ContinuousBatchingStrategy::MergePendingSwapoutRequests(bool blocking, bool early_stop) {
  KLLM_LOG_DEBUG << "MergePendingSwapoutRequests invoked. " << *batch_state_ << ", blocking:" << blocking
                 << ", early_stop:" << early_stop;

  // Wait all requests done.
  size_t swapout_left_req_num = 0;
  do {
    std::vector<int64_t> swapout_req_ids;
    Status status = cache_manager_->WaitSwapoutRequests(swapout_req_ids, swapout_left_req_num, blocking);
    if (!status.OK()) {
      KLLM_LOG_DEBUG << "Error MergePendingSwapoutRequests WaitSwapoutRequests failed. swapout_req_ids:"
                     << swapout_req_ids << ", info: " << status.GetMessage();
      return status;
    }

    KLLM_LOG_DEBUG << "finished swapout request size:" << swapout_req_ids.size();
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
      if (req->attn_dp_group_id == batch_state_->schedule_output->merged_swapout_req_ids.size()) {
        batch_state_->schedule_output->merged_swapout_req_ids.push_back(std::vector<int64_t>{req->req_id});
      } else {
        batch_state_->schedule_output->merged_swapout_req_ids[req->attn_dp_group_id].push_back(req->req_id);
      }

      KLLM_LOG_DEBUG << "after finish swapout " << *req << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << batch_state_;

      batch_state_->swapped_queue[req->req_id] = req;
      batch_state_->swapout_pending_requests_.erase(it);
      KLLM_LOG_DEBUG << "after finish swapout " << *req << *batch_state_;
    }
  } while (!early_stop && swapout_left_req_num > 0);

  return Status();
}

void ContinuousBatchingStrategy::ProcessTransferQueue() {
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
  batch_state_->MergeWaitingReqs(waiting_reqs);
  ProcessRunningQueue();
  ProcessSwappedQueue();
  ProcessWaitingQueue();
  ProcessTransferQueue();

  REPORT_COUNTER(batch_scheduler_pending_swapin_size, batch_state_->swapin_pending_requests_.size());
  REPORT_COUNTER(batch_scheduler_pending_swapout_size, batch_state_->swapout_pending_requests_.size());
}

}  // namespace ksana_llm
