/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <future>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
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

BatchScheduler::BatchScheduler(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                               bool always_return_launchable_req,
                               std::vector<std::shared_ptr<ModelInstance>>& model_instances)
    : batch_scheduler_config_(batch_scheduler_config),
      dp_num_(runtime_config.parallel_basic_config.attn_data_parallel_size),
      always_return_launchable_req_(always_return_launchable_req),
      model_instances_(model_instances) {
  // Config validation.
  KLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_step_token_num >= batch_scheduler_config_.max_token_len,
                       FormatStr("The max_step_token_num must larger or equal than max_token_len, %d vs %d.",
                                 batch_scheduler_config_.max_step_token_num, batch_scheduler_config_.max_token_len));
  pp_batch_num_ = batch_scheduler_config_.max_pp_batch_num > 0 ? batch_scheduler_config_.max_pp_batch_num : 1;

  // max_waiting_queue_len is for each strategy
  waiting_reqs_.reserve(batch_scheduler_config_.max_waiting_queue_len * dp_num_);

  schedule_output_group_ = std::make_shared<ScheduleOutputGroup>(dp_num_);
  KLLM_LOG_DEBUG << "pp_batch_num_=" << pp_batch_num_ << ", batch_scheduler_config_.pp_multibatch_wb_strategy="
                 << batch_scheduler_config_.pp_multibatch_wb_strategy;
  if (batch_scheduler_config_.pp_multibatch_wb_strategy != PPMultibatchWBStrategy::NO_WB) {
    pp_multibatch_wl_balancer_ =
        std::make_unique<PPMultibatchWorkloadBalancer>(batch_scheduler_config_.pp_multibatch_wb_strategy);
  }
  balance_reqs_algo_ = std::make_unique<BalanceReqsAlgo>();
  threadpool_ = std::make_unique<ThreadPool>(dp_num_);
  threadpool_->Start();

  scheduler_shared_counter_ = std::make_shared<SchedulerSharedCounter>(dp_num_);
  scheduler_ticktok_ = std::make_shared<SchedulerTickTok>(dp_num_);

  schedule_strategies_.resize(dp_num_);
  batch_states_.resize(dp_num_);
  dp_waiting_reqs_.resize(dp_num_);

  for (int i = 0; i < dp_num_; i++) {
    schedule_strategies_[i] = ScheduleStrategyFactory::CreateScheduleStrategy(batch_scheduler_config_, runtime_config);
    schedule_strategies_[i]->SetSharedCounter(scheduler_shared_counter_);
    schedule_strategies_[i]->SetSchedulerTickTok(scheduler_ticktok_);
    schedule_strategies_[i]->SetDataParaGroupId(i);

    batch_states_[i].resize(pp_batch_num_);
    for (size_t j = 0; j < pp_batch_num_; j++) {
      batch_states_[i][j] = std::make_shared<BatchState>(j, batch_scheduler_config_);
    }
    dp_waiting_reqs_[i].reserve(batch_scheduler_config_.max_waiting_queue_len);
  }

  // Create mock request to make sure there are always launchable requests.
  if (always_return_launchable_req_) {
    CreateMockReq(runtime_config, mock_request_group_);
    if (mock_request_group_.size() >= 1) {
      for (int i = 0; i < pp_batch_num_; i++) {
        batch_states_[0][i]->mock_queue.push_back(mock_request_group_[0]);
        std::shared_ptr<InferRequest> req = *batch_states_[0][i]->mock_queue.begin();

        KLLM_LOG_DEBUG << "req_id " << req->req_id << ", input_tokens_num " << req->input_tokens.size()
                       << ", output_tokens_num " << req->output_tokens.size() << ", InferRequest addr " << req
                       << ", output_tokens addr" << req->output_tokens.data() << ", input_tokens addr "
                       << req->input_tokens.data();
      }
    }
  }
}

BatchScheduler::~BatchScheduler() { threadpool_->Stop(); }

void BatchScheduler::Stop() { terminating_ = true; }

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

  UpdateQPS(infer_request_group.size());  // 更新QPS统计，传入实际请求数量
  return EnqueueWaitingBufferQueue(infer_request_group);
}

bool BatchScheduler::IsIdle(size_t multi_batch_id) {
  bool waiting_buffer_empty = false;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    waiting_buffer_empty = waiting_reqs_.empty();
  }

  bool batch_state_queue_empty = true;
  for (auto& dp_batch_states : batch_states_) {
    auto& batch_state = dp_batch_states[multi_batch_id];
    std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
    batch_state_queue_empty = batch_state_queue_empty && batch_state->decoding_queue.empty() &&
                              batch_state->waiting_queue.empty() && batch_state->transfer_queue.empty();
  }

  return (waiting_buffer_empty && batch_state_queue_empty);
}

void BatchScheduler::WaitUntilHaveReqs(size_t multi_batch_id) {
  while (IsIdle(multi_batch_id) && !terminating_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ReportTotalState();
  }
}

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
    infer_request->output_tokens = infer_request->input_tokens;
    infer_request->ResetPrefillingTokens();
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
      KLLM_LOG_SCHEDULER << "waiting_reqs_ is empty";
      return;
    }

    if (waiting_reqs_.size() == 1 && dp_waiting_reqs_.size() == 1) {
      dp_waiting_reqs_[0].insert(dp_waiting_reqs_[0].end(), waiting_reqs_.begin(), waiting_reqs_.end());
      waiting_reqs_.clear();
      KLLM_LOG_SCHEDULER << "waiting_reqs_ size is 1";
      return;
    }

    for (auto& req : waiting_reqs_) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        // forwarding_tokens is empty at first time
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    waiting_reqs_.clear();
  }

  std::vector<float> workload(dp_num_, 0);
  for (size_t i = 0; i < dp_num_; ++i) {
    auto& dp_waiting_reqs = dp_waiting_reqs_[i];
    for (auto& req : dp_waiting_reqs) {
      int64_t tokens_num = 0;
      if (req->forwarding_tokens.size() > 0) {
        tokens_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
      } else {
        // forwarding_tokens is empty at first time
        tokens_num = req->input_tokens.size() - req->kv_cached_token_num;
      }
      tokens_num = tokens_num > 0 ? tokens_num : 1;
      waiting_reqs_with_index.emplace_back(
          std::make_pair<size_t, std::shared_ptr<InferRequest>>(static_cast<size_t>(tokens_num), std::move(req)));
    }
    dp_waiting_reqs.clear();

    size_t running_size = 0;
    size_t waiting_size = 0;
    for (int j = 0; j < pp_batch_num_; j++) {
      auto& batch_state = batch_states_[i][j];
      std::lock_guard<std::mutex> guard(batch_state->queue_mutex);

      // Note(TJ): 最好可以使用每个req的tokens总和
      running_size += batch_state->schedule_output->running_reqs.size();
      waiting_size += batch_state->waiting_queue.size();
    }
    // 计算负载，根据优先级分配不同权重，数值越低，权重越低
    workload[i] = running_size * 0.7f + waiting_size;
  }

  balance_reqs_algo_->BalanceReqs(workload, waiting_reqs_with_index, dp_waiting_reqs_);
}

void BatchScheduler::BalancePPMultiBatchReqs(size_t multi_batch_id) {
  if (!pp_multibatch_wl_balancer_) return;

  for (size_t i = 0; i < dp_num_; ++i) {
    pp_multibatch_wl_balancer_->BalancePPMultiBatchReqs(multi_batch_id, dp_waiting_reqs_[i], batch_states_[i]);
  }
}

void BatchScheduler::ReportBatchState(std::shared_ptr<BatchState> batch_state, std::string dp_idx_str) {
  size_t batch_size = batch_state->schedule_output->running_reqs.size();
  REPORT_METRIC("batch_scheduler_batch_size_" + dp_idx_str, batch_size);
  REPORT_METRIC("batch_scheduler_waiting_size_" + dp_idx_str, batch_state->waiting_queue.size());
  REPORT_METRIC("batch_scheduler_batch_size", batch_size);
  REPORT_METRIC("batch_scheduler_waiting_size", batch_state->waiting_queue.size());

  if (batch_size > 0) {
    size_t token_num = 0;
    const auto current_time = ProfileTimer::GetCurrentTimeInUs();
    for (const auto& req : batch_state->schedule_output->running_reqs) {
      token_num += req->forwarding_tokens.size();
      if (req->kv_cached_token_num == 0) {
        REPORT_METRIC("batch_manager_schedule_us", current_time - req->timestamp_in_us);
      }
    }
    REPORT_METRIC("num_tokens_to_schedule", token_num);
  }
}

void BatchScheduler::UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output) {
  std::lock_guard<std::mutex> guard(schedule_mutex_);
  // Update running requests before workload balance
  for (size_t i = 0; i < dp_num_; i++) {
    schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
    schedule_strategies_[i]->UpdateRunningRequests(generation_output.reqs[i]);
  }
}

std::shared_ptr<ScheduleOutputGroup> BatchScheduler::Schedule(size_t multi_batch_id) {
  PROFILE_EVENT_SCOPE(Schedule_, fmt::format("Schedule_{}", multi_batch_id));
  std::lock_guard<std::mutex> guard(schedule_mutex_);

  KLLM_LOG_DEBUG << "Try scheduler multi_batch_id=" << multi_batch_id << ", waiting_reqs_size:" << waiting_reqs_.size();
  Singleton<LayerProgressTracker>::GetInstance()->ResetState();

  BalanceWaitingReqs();

  BalancePPMultiBatchReqs(multi_batch_id);

  std::vector<std::future<void>> futures;
  for (size_t i = 0; i < dp_num_; i++) {
    futures.push_back(threadpool_->Submit([this, i, multi_batch_id] {
      schedule_strategies_[i]->SetBatchState(batch_states_[i][multi_batch_id]);
      schedule_strategies_[i]->Schedule(dp_waiting_reqs_[i]);
    }));
  }

  for (auto& future : futures) {
    future.wait();
  }

  size_t total_running_size = 0;
  size_t total_waiting_size_in_batch_states = 0;
  size_t total_dp_waiting_queue_size = 0;
  for (size_t i = 0; i < dp_num_; i++) {
    auto& batch_state = batch_states_[i][multi_batch_id];
    ReportBatchState(batch_state, std::to_string(i));
    schedule_output_group_->outputs[i] = batch_state->schedule_output;
    total_running_size += batch_state->schedule_output->running_reqs.size();
    total_waiting_size_in_batch_states += batch_state->waiting_queue.size();
    total_dp_waiting_queue_size += dp_waiting_reqs_[i].size();
  }

  if (always_return_launchable_req_ && total_running_size == 0) {
    for (size_t i = 0; i < dp_num_; i++) {
      auto& batch_state = batch_states_[i][multi_batch_id];
      if (!batch_state->mock_queue.empty()) {
        auto it = batch_state->mock_queue.begin();
        batch_state->waiting_queue.push_back(*it);
        batch_state->mock_queue.erase(it);
      }
    }
  }
  // TODO(xingjinglu): remove potential mock request when running_size > 1.

  schedule_output_group_->schedule_id++;

  KLLM_LOG_SCHEDULER << "Finish schedule. multi_batch_id=" << multi_batch_id
                     << ", schedule_id=" << schedule_output_group_->schedule_id
                     << ", running_req.size(): " << total_running_size
                     << ", total_waiting_size_in_batch_states=" << total_waiting_size_in_batch_states
                     << ", total_dp_waiting_queue_size=" << total_dp_waiting_queue_size;

  ReportTotalState();
  return schedule_output_group_;
}

void BatchScheduler::ReportTotalState() {
  static time_t last_report_time_ms = ProfileTimer::GetCurrentTimeInMs();
  time_t current_time_ms = ProfileTimer::GetCurrentTimeInMs();
  // Report every 10 seconds.
  constexpr size_t kReportIntervalMs = 10000;
  if ((current_time_ms - last_report_time_ms) < kReportIntervalMs) {
    return;
  }
  last_report_time_ms = current_time_ms;

  size_t total_running_size = 0;
  size_t total_waiting_size = 0;
  size_t total_transfer_size = 0;
  {
    std::lock_guard<std::mutex> guard(waiting_reqs_mutex_);
    total_waiting_size = waiting_reqs_.size();
    for (size_t dp_rank = 0; dp_rank < dp_num_; ++dp_rank) {
      total_waiting_size += dp_waiting_reqs_[dp_rank].size();
      auto& batch_states = batch_states_[dp_rank];
      for (size_t multi_batch_id = 0; multi_batch_id < pp_batch_num_; ++multi_batch_id) {
        auto& batch_state = batch_states[multi_batch_id];
        std::lock_guard<std::mutex> guard(batch_state->queue_mutex);
        total_running_size += batch_state->schedule_output->running_reqs.size();
        total_waiting_size += batch_state->waiting_queue.size();
        total_transfer_size += batch_state->transfer_queue.size();
      }
    }
  }

  size_t total_used_blocks_num = 0;
  size_t total_free_blocks_num = 0;

  for (size_t dp_rank = 0; dp_rank < dp_num_; ++dp_rank) {
    auto& cache_manager = schedule_strategies_[dp_rank]->GetCacheManager();
    total_used_blocks_num += cache_manager->GetUsedBlockNumber();
    total_free_blocks_num += cache_manager->GetUsableBlockNumber();
  }
  size_t total_block_num = total_used_blocks_num + total_free_blocks_num;
  KLLM_LOG_INFO << " running_req_num=" << total_running_size << ", waiting_req_num=" << total_waiting_size
                << ", transfer_req_num=" << total_transfer_size << ", free_block_num=" << total_free_blocks_num
                << ", block_utils=" << (total_used_blocks_num * 100 / total_block_num) << "%";
}

Status BatchScheduler::CreateMockReq(const RuntimeConfig& runtime_config,
                                     std::vector<std::shared_ptr<InferRequest>>& infer_request_group) {
  // To avoid Mock requests being categorized as SingleTokenForward requests, we calculate the Mock request total
  // length as: Mock total length = MTP token count + SingleToken length + 1 (additional token)
  size_t mock_req_length = (runtime_config.enable_mtp_module ? 1 : 0) + 1 + 1;
  auto mock_req_input = std::make_shared<KsanaPythonInput>();
  alias_python_input_ = mock_req_input;
  std::vector<int> input_tokens(mock_req_length, 0);
  for (int i = 0; i < mock_req_length; ++i) {
    input_tokens[i] = (i + 1) % 100;  // Fill with some dummy tokens.
  }

  mock_req_input->input_tokens = input_tokens;
  // Only do one token prefill.
  mock_req_input->sampling_config.max_new_tokens = 1;
  mock_req_input->sampling_config.ignore_eos = true;
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto mock_req = std::make_shared<Request>(mock_req_input, req_ctx);
  alias_mock_request_ = mock_req;

  mock_req->waiter = std::make_shared<Waiter>(1);
  KLLM_LOG_DEBUG << "mock_req req_id " << mock_req->req_id << ", input_tokens_num " << mock_req->input_tokens.size()
                 << ", output_tokens addr " << mock_req->output_tokens.data();

  // mock_req->output_group.size() == 1.
  for (size_t i = 0; i < mock_req->output_group.size(); i++) {
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(mock_req, i);
    infer_request_group.push_back(infer_req);
    RuntimeConfig runtime_config;
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
    infer_req->kv_cache_blocks.resize(runtime_config.parallel_basic_config.attn_tensor_parallel_size);
    CacheManagerConfig cache_manager_config;
    Singleton<Environment>::GetInstance()->GetCacheManagerConfig(cache_manager_config);
    infer_req->block_token_num = cache_manager_config.block_token_num;
    infer_req->model_instance = model_instances_[0];
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;
    infer_req->step = 0;
    infer_req->kv_cached_token_num = 0;
    infer_req->req_id = mock_req->req_id;
    infer_req->is_mock_req = true;
  }

  for (auto& infer_req : infer_request_group) {
    infer_req->SetReqGroup(infer_request_group);
    KLLM_LOG_DEBUG << "InferRequest output_tokens_num " << infer_req->output_tokens.size() << ", Addr " << infer_req
                   << ", output_tokens addr " << infer_req->output_tokens.data();
  }

  return Status();
}

}  // namespace ksana_llm
