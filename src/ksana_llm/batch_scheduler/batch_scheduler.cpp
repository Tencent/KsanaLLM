/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/profiler/sched_event_tracer.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BatchScheduler::BatchScheduler(const BatchSchedulerConfig& batch_scheduler_config, int dp_num, int tp_num)
    : batch_scheduler_config_(batch_scheduler_config), dp_num_(dp_num) {
  // Config validation.
  KLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_step_token_num >= batch_scheduler_config_.max_token_len,
                       FormatStr("The max_step_token_num must larger or equal than max_token_len, %d vs %d.",
                                 batch_scheduler_config_.max_step_token_num, batch_scheduler_config_.max_token_len));
  pp_batch_num_ = batch_scheduler_config_.max_pp_batch_num > 0 ? batch_scheduler_config_.max_pp_batch_num : 1;

  // max_waiting_queue_len is for each strategy
  waiting_reqs_.reserve(batch_scheduler_config_.max_waiting_queue_len * dp_num);

  schedule_output_group_ = std::make_shared<ScheduleOutputGroup>(dp_num);
  KLLM_LOG_DEBUG << "pp_batch_num_=" << pp_batch_num_ << ", batch_scheduler_config_.pp_multibatch_wb_strategy="
                 << batch_scheduler_config_.pp_multibatch_wb_strategy;
  if (batch_scheduler_config_.pp_multibatch_wb_strategy != PPMultibatchWBStrategy::NO_WB) {
    pp_multibatch_wl_balancer_ =
        std::make_unique<PPMultibatchWorkloadBalancer>(batch_scheduler_config_.pp_multibatch_wb_strategy);
  }
  balance_reqs_algo_ = std::make_unique<BalanceReqsAlgo>();
  threadpool_ = std::make_unique<ThreadPool>(dp_num);
  threadpool_->Start();

  schedule_strategies_.resize(dp_num);
  batch_states_.resize(dp_num);
  dp_waiting_reqs_.resize(dp_num);

  for (int i = 0; i < dp_num; i++) {
    schedule_strategies_[i] = ScheduleStrategyFactory::CreateScheduleStrategy(batch_scheduler_config_);
    batch_states_[i].resize(pp_batch_num_);
    for (size_t j = 0; j < pp_batch_num_; j++) {
      batch_states_[i][j] = std::make_shared<BatchState>(j, batch_scheduler_config_);
    }
    dp_waiting_reqs_[i].reserve(batch_scheduler_config_.max_waiting_queue_len);
  }
}

BatchScheduler::~BatchScheduler() {
  Stop();
  threadpool_->Stop();
}

void BatchScheduler::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int dp_idx) {
  KLLM_CHECK_WITH_INFO(dp_idx < dp_num_, FormatStr("dp_idx %d is out of range, dp_num_ %zu.", dp_idx, dp_num_));
  schedule_strategies_.at(dp_idx)->SetCacheManager(cache_manager);
}

std::shared_ptr<CacheManagerInterface>& BatchScheduler::GetCacheManager(int dp_idx) {
  KLLM_CHECK_WITH_INFO(dp_idx < dp_num_, FormatStr("dp_idx %d is out of range, dp_num_ %zu.", dp_idx, dp_num_));
  return schedule_strategies_[dp_idx]->GetCacheManager();
}

Status BatchScheduler::AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
  KLLM_LOG_DEBUG << "batch scheduler add infer req " << infer_request->req_id << ", max_new_tokens "
                 << infer_request->sampling_config.max_new_tokens;

  if (CheckRequestExceedLength(infer_request)) {
    KLLM_LOG_ERROR << "req_id: " << infer_request->req_id
                   << "input len or logits_custom_length is too long inference failed.";

    const auto finish_status =
        Status(RET_INPUT_LENGTH_EXCEEDED, "input length or logits_custom_length exceeds the limit.");
    infer_request->finish_status = finish_status;
    for (auto& infer_request : infer_request_group) {
      infer_request->finished = true;
    }
    infer_request->Notify();
    return finish_status;
  }

  return EnqueueWaitingBufferQueue(infer_request_group);
}

bool BatchScheduler::IsIdle(size_t multi_batch_id) {
  bool waiting_buffer_emtpy = false;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    waiting_buffer_emtpy = waiting_reqs_.empty();
  }

  bool batch_state_queue_empty = true;
  for (auto& dp_batch_states : batch_states_) {
    auto& batch_state = dp_batch_states[multi_batch_id];
    std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
    batch_state_queue_empty = batch_state_queue_empty && batch_state->swapped_queue.empty() &&
                              batch_state->waiting_queue.empty() && batch_state->transfer_queue.empty();
  }

  return (waiting_buffer_emtpy && batch_state_queue_empty);
}

void BatchScheduler::WaitUntilHaveReqs(size_t multi_batch_id) {
  while (IsIdle(multi_batch_id) && !terminating_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    {
      std::lock_guard<std::mutex> guard(schedule_mutex_);
      // Update requests in swapin/swapout pending queue
      for (size_t i = 0; i < dp_num_; i++) {
        auto batch_state = batch_states_[i][multi_batch_id];
        if (batch_state->swapin_pending_requests_.empty() && batch_state->swapout_pending_requests_.empty()) {
          continue;
        }
        schedule_strategies_[i]->SetBatchState(batch_state);
        schedule_strategies_[i]->UpdateSwapPendingRequests();
      }
    }
  }
}

void BatchScheduler::Stop() { terminating_ = true; }

Status BatchScheduler::EnqueueWaitingBufferQueue(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);

  if (waiting_reqs_.size() + infer_request_group.size() > batch_scheduler_config_.max_waiting_queue_len) {
    std::shared_ptr<InferRequest>& infer_request = infer_request_group[0];
    KLLM_LOG_ERROR << "waiting queue is full, req " << infer_request << " failed."
                   << " waiting queue size: " << waiting_reqs_.size()
                   << ", max_waiting_queue_len: " << batch_scheduler_config_.max_waiting_queue_len
                   << ", infer_request_group_size: " << infer_request_group.size();

    auto finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
    infer_request->finish_status = finish_status;
    for (auto& req : infer_request_group) {
      req->finished = true;
    }
    infer_request->Notify();
    return finish_status;
  }

  for (const auto& infer_request : infer_request_group) {
    waiting_reqs_.push_back(infer_request);
  }
  return Status();
}

inline bool BatchScheduler::CheckRequestExceedLength(const std::shared_ptr<InferRequest> req) {
  return req->input_tokens.size() > batch_scheduler_config_.max_token_len ||
         req->logits_custom_length > std::min(req->input_tokens.size(), batch_scheduler_config_.max_batch_size);
}

void BatchScheduler::BalanceWaitingReqs() {
  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>> waiting_reqs_with_index;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    // inputs are waiting_reqs_ and batch_states_
    // output is dp_waiting_reqs_
    if (waiting_reqs_.empty()) {
      return;
    }

    if (dp_waiting_reqs_.size() == 1) {
      dp_waiting_reqs_[0].insert(dp_waiting_reqs_[0].end(), waiting_reqs_.begin(), waiting_reqs_.end());
      waiting_reqs_.clear();
      return;
    }

    for (auto& req : waiting_reqs_) {
      int64_t tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    waiting_reqs_.clear();
  }

  std::vector<float> workload(dp_num_, 0);
  for (size_t i = 0; i < dp_num_; ++i) {
    auto& waiting_reqs = dp_waiting_reqs_[i];
    for (auto& req : waiting_reqs) {
      int64_t tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    waiting_reqs.clear();

    size_t running_size = 0;
    size_t swapped_size = 0;
    size_t waiting_size = 0;
    for (int j = 0; j < pp_batch_num_; j++) {
      auto& batch_state = batch_states_[i][j];
      std::lock_guard<std::mutex> guard(batch_state->queue_mutex);

      running_size += batch_state->schedule_output->running_reqs.size();
      swapped_size += batch_state->swapped_queue.size();
      waiting_size += batch_state->waiting_queue.size();
    }
    // 计算负载，根据优先级分配不同权重，数值越低，权重越低
    workload[i] = running_size * 0.7f + waiting_size + swapped_size * 1.6f;
  }

  balance_reqs_algo_->BalanceReqs(workload, waiting_reqs_with_index, dp_waiting_reqs_);
}

void BatchScheduler::BalancePPMultiBatchReqs(size_t multi_batch_id) {
  if (!pp_multibatch_wl_balancer_) return;

  for (size_t i = 0; i < dp_num_; ++i) {
    pp_multibatch_wl_balancer_->BalancePPMultiBatchReqs(multi_batch_id, dp_waiting_reqs_[i], batch_states_[i]);
  }
}

void BatchScheduler::ReportBatchState(std::shared_ptr<BatchState> batch_state) {
  size_t batch_size = batch_state->schedule_output->running_reqs.size();
  REPORT_METRIC(batch_scheduler_batch_size, batch_size);
  REPORT_METRIC(batch_scheduler_waiting_size, batch_state->waiting_queue.size());
  REPORT_METRIC(batch_scheduler_swapped_size, batch_state->swapped_queue.size());

  if (batch_size > 0) {
    size_t token_num = 0;
    const auto current_time = ProfileTimer::GetCurrentTimeInMs();
    for (const auto& req : batch_state->schedule_output->running_reqs) {
      token_num += req->forwarding_tokens.size();
      if (req->kv_cached_token_num == 0) {
        REPORT_METRIC(batch_manager_schedule_ms, current_time - req->timestamp_in_ms);
      }
      REPORT_METRIC(req_total_cost_in_queue_ms, current_time - req->timestamp_in_ms);
    }
    REPORT_METRIC(token_num_in_batch, token_num);
  }
}

std::shared_ptr<ScheduleOutputGroup> BatchScheduler::Schedule(size_t multi_batch_id) {
  std::lock_guard<std::mutex> guard(schedule_mutex_);

  KLLM_LOG_DEBUG << "Try scheduler multi_batch_id=" << multi_batch_id << ", waiting_reqs_size:" << waiting_reqs_.size();
  Singleton<LayerProgressTracker>::GetInstance()->ResetState();

  // Update running requests before workload balance
  for (size_t i = 0; i < dp_num_; i++) {
    schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
    schedule_strategies_[i]->UpdateRunningRequests();
  }

  BalanceWaitingReqs();

  BalancePPMultiBatchReqs(multi_batch_id);

  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < dp_num_; i++) {
    futures.push_back(
        threadpool_->Submit([this, i, multi_batch_id] { schedule_strategies_[i]->Schedule(dp_waiting_reqs_[i]); }));
  }

  for (auto& future : futures) {
    future.wait();
  }

  size_t total_running_size = 0;
  size_t total_waiting_size_in_batch_states = 0;
  size_t total_dp_waiting_queue_size = 0;
  for (size_t i = 0; i < dp_num_; i++) {
    auto& batch_state = batch_states_[i][multi_batch_id];
    ReportBatchState(batch_state);
    schedule_output_group_->outputs[i] = batch_state->schedule_output;
    total_running_size += batch_state->schedule_output->running_reqs.size();
    total_waiting_size_in_batch_states += batch_state->waiting_queue.size();
    total_dp_waiting_queue_size += dp_waiting_reqs_[i].size();
  }
  schedule_output_group_->schedule_id++;

  KLLM_LOG_DEBUG << "Finish schedule. multi_batch_id=" << multi_batch_id
                 << ", schedule_id=" << schedule_output_group_->schedule_id
                 << ", running_req.size(): " << total_running_size
                 << ", total_waiting_size_in_batch_states=" << total_waiting_size_in_batch_states
                 << ", total_dp_waiting_queue_size=" << total_dp_waiting_queue_size;
  return schedule_output_group_;
}

}  // namespace ksana_llm
