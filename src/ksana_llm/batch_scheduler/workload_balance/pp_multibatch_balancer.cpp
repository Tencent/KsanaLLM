/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/workload_balance/pp_multibatch_balancer.h"

namespace ksana_llm {

PPMultibatchWorkloadBalancer::PPMultibatchWorkloadBalancer(PPMultibatchWBStrategy pp_multibatch_wb_strategy)
    : pp_multibatch_wb_strategy_(pp_multibatch_wb_strategy) {
  threshold_factor_ = 1.2f;
  KLLM_LOG_INFO << "PPMultibatchWorkloadBalancer init with strategy: " << pp_multibatch_wb_strategy
                << ", threshold_factor_=" << threshold_factor_;
  KLLM_CHECK_WITH_INFO((pp_multibatch_wb_strategy >= 0) && (pp_multibatch_wb_strategy <= 4),
                       FormatStr("invalid pp_multibatch_wb_strategy: %d ", pp_multibatch_wb_strategy));
}

void PPMultibatchWorkloadBalancer::BalancePPMultiBatchReqs(size_t multi_batch_id,
                                                           std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                                                           std::vector<std::shared_ptr<BatchState>>& batch_states) {
  // First distribute waiting_reqs to batch states to balance workload
  DistributeWaitingReqs(waiting_reqs, batch_states);
  if (pp_multibatch_wb_strategy_ != PPMultibatchWBStrategy::NO_DYNAMIC_WB) {
    OffloadBatchWorkload(multi_batch_id, batch_states);
  }
}

void PPMultibatchWorkloadBalancer::DistributeWaitingReqs(std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                                                         std::vector<std::shared_ptr<BatchState>>& batch_states) {
  if (waiting_reqs.empty() || batch_states.empty()) {
    return;
  }

  KLLM_LOG_SCHEDULER << "Distributing " << waiting_reqs.size() << " waiting requests to balance workload";

  // Calculate the current workload for each batch state
  std::vector<size_t> current_workloads;
  size_t total_current_workload = 0;

  for (const auto& batch : batch_states) {
    // Calculate workload using CalculateWorkload for each request
    size_t batch_workload = 0;

    // Calculate workload for decoding_queue (vector)
    batch_workload += CalculateWorkloadForReqs(batch->decoding_queue);

    // Calculate workload for waiting_queue (deque) by directly using CalculateWorkload
    for (const auto& req : batch->waiting_queue) {
      batch_workload += CalculateWorkload(req);
    }

    current_workloads.push_back(batch_workload);
    total_current_workload += batch_workload;
  }

  // Greedy approach: distribute each waiting request to the batch with minimum workload
  std::stringstream ss;
  std::vector<std::vector<int>> req_list{batch_states.size()};
  for (auto& req : waiting_reqs) {
    auto min_it = std::min_element(current_workloads.begin(), current_workloads.end());
    int min_idx = std::distance(current_workloads.begin(), min_it);

    auto& min_batch = batch_states[min_idx];
    {
      std::lock_guard<std::mutex> guard(min_batch->queue_mutex);
      min_batch->waiting_queue.push_back(req);
    }
    size_t cur_req_workload = CalculateWorkload(req);
    KLLM_LOG_SCHEDULER << "req_id:" << req->req_id << ", cur_req_workload:" << cur_req_workload << ", add to batch id "
                       << min_idx << ", last batch workload:" << current_workloads[min_idx];
    req_list[min_idx].push_back(req->req_id);
    current_workloads[min_idx] += cur_req_workload;
  }

  for (size_t i = 0; i < req_list.size(); i++) {
    auto& reqs = req_list[i];
    if (reqs.empty()) {
    } else {
      ss << "[ batch " << batch_states[i]->multi_batch_id_ << " add " << reqs.size()
         << " reqs, workload: " << current_workloads[i] << ", req ids: ";
      for (auto req_id : reqs) {
        ss << req_id << ", ";
      }
      ss << " ] ";
    }
  }
  KLLM_LOG_SCHEDULER << "Distribute " << waiting_reqs.size() << " reqs. " << ss.str();

  ss.str("");
  for (size_t i = 0; i < batch_states.size(); i++) {
    auto& batch = batch_states[i];
    ss << "[ batch " << batch->multi_batch_id_ << ", workload:" << current_workloads[i]
       << ", decoding:" << batch->decoding_queue.size() << ", waiting:" << batch->waiting_queue.size() << " ] ";
  }
  KLLM_LOG_SCHEDULER << "After distribute: " << ss.str();

  // Clear the waiting_reqs vector since all requests have been
  // distributed
  waiting_reqs.clear();
}

void PPMultibatchWorkloadBalancer::DistributeDecodingReqs(std::vector<std::shared_ptr<InferRequest>>& decoding_reqs,
                                                          std::vector<std::shared_ptr<BatchState>>& batch_states) {
  if (decoding_reqs.empty() || batch_states.empty()) {
    return;
  }

  KLLM_LOG_SCHEDULER << "Distributing " << decoding_reqs.size() << " decoding requests to balance workload";

  // Calculate the current workload for each batch state
  std::vector<size_t> current_workloads;
  size_t total_current_workload = 0;

  for (const auto& batch : batch_states) {
    // Calculate workload using CalculateWorkload for each request
    size_t batch_workload = 0;

    // Calculate workload for decoding_queue (vector)
    batch_workload += CalculateWorkloadForReqs(batch->decoding_queue);

    current_workloads.push_back(batch_workload);
    total_current_workload += batch_workload;
  }

  // Greedy approach: distribute each decoding request to the batch with minimum workload
  std::stringstream ss;
  std::vector<std::vector<int>> req_list{batch_states.size()};
  for (auto& req : decoding_reqs) {
    auto min_it = std::min_element(current_workloads.begin(), current_workloads.end());
    int min_idx = std::distance(current_workloads.begin(), min_it);

    auto& min_batch = batch_states[min_idx];
    {
      std::lock_guard<std::mutex> guard(min_batch->queue_mutex);
      min_batch->decoding_queue.push_back(req);
    }
    size_t cur_req_workload = CalculateWorkload(req);
    KLLM_LOG_SCHEDULER << "req_id:" << req->req_id << ", cur_req_workload:" << cur_req_workload << ", add to batch id "
                       << min_idx << ", last batch workload:" << current_workloads[min_idx];
    req_list[min_idx].push_back(req->req_id);
    current_workloads[min_idx] += cur_req_workload;
  }

  for (size_t i = 0; i < req_list.size(); i++) {
    auto& reqs = req_list[i];
    if (reqs.empty()) {
    } else {
      ss << "[ batch " << batch_states[i]->multi_batch_id_ << " add " << reqs.size()
         << " reqs, workload: " << current_workloads[i] << ", req ids: ";
      for (auto req_id : reqs) {
        ss << req_id << ", ";
      }
      ss << " ] ";
    }
  }
  KLLM_LOG_SCHEDULER << "Distribute " << decoding_reqs.size() << " reqs. " << ss.str();

  ss.str("");
  for (size_t i = 0; i < batch_states.size(); i++) {
    auto& batch = batch_states[i];
    ss << "[ batch " << batch->multi_batch_id_ << ", workload:" << current_workloads[i]
       << ", decoding:" << batch->decoding_queue.size() << ", waiting:" << batch->waiting_queue.size() << " ] ";
  }
  KLLM_LOG_SCHEDULER << "After distribute: " << ss.str();

  // Clear the decoding_reqs vector since all requests have been distributed
  decoding_reqs.clear();
}

void PPMultibatchWorkloadBalancer::OffloadBatchWorkload(size_t multi_batch_id,
                                                        std::vector<std::shared_ptr<BatchState>>& batch_states) {
  if (batch_states.size() <= 1) {
    // Nothing to balance with only one batch state
    return;
  }
  // Get the target batch state
  auto& target_batch = batch_states[multi_batch_id];

  // No need to offload if few requests left
  if ((target_batch->decoding_queue.size() + target_batch->waiting_queue.size()) < 5) {
    return;
  }

  // Calculate the workload for each batch state using CalculateWorkloadForReqs
  std::vector<size_t> workloads;
  size_t total_workload = 0;

  int target_decoding_workload, target_waiting_workload;
  for (size_t i = 0; i < batch_states.size(); i++) {
    auto& batch = batch_states[i];
    size_t batch_workload = 0;

    // Calculate workload for decoding_queue (vector)
    int decoding_workload = CalculateWorkloadForReqs(batch->decoding_queue);
    batch_workload += decoding_workload;

    // Calculate workload for waiting_queue (deque) by directly using CalculateWorkload
    int waiting_workload = 0;
    for (const auto& req : batch->waiting_queue) {
      waiting_workload += CalculateWorkload(req);
    }
    batch_workload = decoding_workload + waiting_workload;
    if (i == multi_batch_id) {
      target_decoding_workload = decoding_workload;
      target_waiting_workload = waiting_workload;
    }
    workloads.push_back(batch_workload);
    total_workload += batch_workload;
  }

  // Calculate the ideal workload per batch
  size_t ideal_workload = total_workload / batch_states.size();
  if (total_workload % batch_states.size() != 0) {
    ideal_workload += 1;  // Round up to ensure we don't exceed capacity
  }

  // Define a threshold (e.g., 20% more than ideal) to determine if balancing is needed
  size_t threshold_workload = static_cast<size_t>(ideal_workload * threshold_factor_);

  // Check if target batch's workload exceeds the threshold
  if (workloads[multi_batch_id] <= threshold_workload) {
    KLLM_LOG_SCHEDULER << "No need to balance workload. Target batch " << multi_batch_id << " has workload "
                       << workloads[multi_batch_id] << "(decoding: " << target_decoding_workload
                       << ", waiting:" << target_waiting_workload << "), which is below threshold "
                       << threshold_workload;
    return;
  }

  // Calculate how much workload needs to be moved from the target batch
  size_t workload_to_move = workloads[multi_batch_id] - ideal_workload;

  KLLM_LOG_SCHEDULER << "Balancing workload. Target batch " << multi_batch_id << " has workload "
                     << workloads[multi_batch_id] << "( decoding: " << target_decoding_workload
                     << ", waiting:" << target_waiting_workload << " ), ideal_workload: " << ideal_workload
                     << ", threshold_workload: " << threshold_workload << ", workload_to_move: " << workload_to_move;

  // Lock the target batch's queue mutex to safely access its queues
  std::lock_guard<std::mutex> target_guard(target_batch->queue_mutex);

  // Collect requests to migrate
  std::vector<std::shared_ptr<InferRequest>> waiting_requests_to_migrate, decoding_requests_to_migrate;
  size_t migrated_workload = 0;

  // First try to migrate requests from waiting_queue
  auto waiting_it = target_batch->waiting_queue.begin();
  while (waiting_it != target_batch->waiting_queue.end() && migrated_workload < workload_to_move) {
    int req_workload = CalculateWorkload(*waiting_it);
    if ((migrated_workload + req_workload) > workload_to_move) {
      // TODO(robertyuan): find better requests
      break;
    }
    waiting_requests_to_migrate.push_back(*waiting_it);
    migrated_workload += req_workload;
    target_waiting_workload -= req_workload;
    waiting_it = target_batch->waiting_queue.erase(waiting_it);
  }

  // If waiting_queue is empty but we still need to move more workload, take from decoding_queue (from the end to
  // minimize impact)
  if (target_batch->waiting_queue.empty() && migrated_workload < workload_to_move &&
      !target_batch->decoding_queue.empty()) {
    auto decoding_it = target_batch->decoding_queue.end() - 1;
    while (decoding_it >= target_batch->decoding_queue.begin() && migrated_workload < workload_to_move) {
      int req_workload = CalculateWorkload(*decoding_it);
      if ((migrated_workload + req_workload) > workload_to_move) {
        // TODO(robertyuan): find better requests
        break;
      }
      decoding_requests_to_migrate.push_back(*decoding_it);
      migrated_workload += req_workload;
      target_waiting_workload -= req_workload;

      // Move iterator before erasing
      auto to_erase = decoding_it;
      decoding_it--;
      target_batch->decoding_queue.erase(to_erase);
    }
  }

  // No request to migrate
  if (decoding_requests_to_migrate.empty() && waiting_requests_to_migrate.empty()) {
    return;
  }

  // Create a vector of batch states excluding the target batch
  std::vector<std::shared_ptr<BatchState>> recipient_batches;
  for (size_t i = 0; i < batch_states.size(); ++i) {
    if (i != multi_batch_id) {
      recipient_batches.push_back(batch_states[i]);
    }
  }

  KLLM_LOG_SCHEDULER << "Migrating waiting_reqs.size=" << waiting_requests_to_migrate.size()
                     << ", decoding_reqs.size=" << decoding_requests_to_migrate.size() << "  with total workload "
                     << migrated_workload << " from batch " << multi_batch_id
                     << ", target workload: (decoding:" << target_decoding_workload
                     << ", waiting:" << target_waiting_workload << " )";
  if (!decoding_requests_to_migrate.empty()) {
    DistributeDecodingReqs(decoding_requests_to_migrate, recipient_batches);
  }

  if (!waiting_requests_to_migrate.empty()) {
    DistributeWaitingReqs(waiting_requests_to_migrate, recipient_batches);
  }
}

int PPMultibatchWorkloadBalancer::CalculateWorkload(const std::shared_ptr<InferRequest>& req) {
  if (!req) {
    return 0;
  }
  int tokens = 0;
  switch (pp_multibatch_wb_strategy_) {
    case PPMultibatchWBStrategy::NO_DYNAMIC_WB:
      return 1;
    case PPMultibatchWBStrategy::WB_BATCH_REQ:
      return 1;
    case PPMultibatchWBStrategy::WB_BATCH_TOKEN:
      // if request is forwarding multiple token, assume [1,200) tokens cost same
      return req->forwarding_tokens.size() / 200 + 1;
    case PPMultibatchWBStrategy::WB_REQ_TOKEN:
      tokens = req->output_tokens.size() > req->kv_cached_token_num
                   ? static_cast<int>(req->output_tokens.size() - req->kv_cached_token_num)
                   : 1;
      return tokens;
    default:
      return 1;
  }
}

int PPMultibatchWorkloadBalancer::CalculateWorkloadForReqs(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  int workload = 0;
  for (auto& req : reqs) {
    workload += CalculateWorkload(req);
  }
  return workload;
}

}  // namespace ksana_llm
