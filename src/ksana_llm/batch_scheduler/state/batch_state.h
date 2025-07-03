/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

struct BatchState {
  explicit BatchState(size_t multi_batch_id, const BatchSchedulerConfig& batch_scheduler_config)
      : multi_batch_id_(multi_batch_id), batch_scheduler_config_(batch_scheduler_config) {
    schedule_output = GetScheduleOutputPool()->GetScheduleOutput();

    schedule_output->running_reqs.reserve(batch_scheduler_config_.max_batch_size);
  }

  void MergeWaitingReqs(std::vector<std::shared_ptr<InferRequest>>& waiting_reqs) {
    std::lock_guard<std::mutex> guard(queue_mutex);

    size_t in_processing_req_num = schedule_output->running_reqs.size();
    in_processing_req_num += waiting_queue.size();
    in_processing_req_num += swapped_queue.size();
    in_processing_req_num += swapin_pending_requests_.size();
    in_processing_req_num += swapout_pending_requests_.size();

    // Process requests from the head of waiting_reqs until we reach max_batch_size
    size_t processed_count = 0;
    while (processed_count < waiting_reqs.size() && in_processing_req_num < batch_scheduler_config_.max_batch_size) {
      auto& infer_request = waiting_reqs[processed_count];
      if (waiting_queue.size() < batch_scheduler_config_.max_waiting_queue_len) {
        waiting_queue.push_back(infer_request);
        in_processing_req_num++;
        processed_count++;
      } else {
        KLLM_LOG_ERROR << "waiting queue is full, req " << infer_request->req_id << " failed.";

        // Reject this request.
        infer_request->finish_status = Status(RET_EXCEED_CAPACITY, "waiting queue is full.");
        infer_request->finished = true;

        infer_request->Notify();
        processed_count++;
      }
    }
    KLLM_LOG_DEBUG << "multi_batch_id: " << multi_batch_id_ << " : Merged " << processed_count << " waiting requests";
    // Remove the processed requests from waiting_reqs
    if (processed_count > 0) {
      waiting_reqs.erase(waiting_reqs.begin(), waiting_reqs.begin() + processed_count);
    }
  }

  void MergeRunningPendingReqs(size_t max_batch_size) {
    std::lock_guard<std::mutex> guard(queue_mutex);

    // Calculate how many requests we can add without exceeding max_batch_size
    size_t current_size = schedule_output->running_reqs.size();
    size_t available_slots = 0;

    if (current_size < max_batch_size) {
      available_slots = max_batch_size - current_size;
    }

    // If we have available slots and pending requests
    if (available_slots > 0 && !running_pending_reqs.empty()) {
      // Determine how many requests to move
      size_t requests_to_move = std::min(available_slots, running_pending_reqs.size());

      // Insert only the calculated number of requests
      schedule_output->running_reqs.insert(schedule_output->running_reqs.end(), running_pending_reqs.begin(),
                                           running_pending_reqs.begin() + requests_to_move);

      // Remove the moved requests from running_pending_reqs
      running_pending_reqs.erase(running_pending_reqs.begin(), running_pending_reqs.begin() + requests_to_move);
    }
  }

  void ResetInfoBeforeSchedule() {
    schedule_time_in_ms = GetCurrentTimeInMs();
    step_sched_finish = false;

    // Reset all swap info.
    schedule_output->Reset();
    schedule_output->multi_batch_id += 1;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << " BatchState(multi_batch_id:" << multi_batch_id_
        << ", running_queue_size:" << schedule_output->running_reqs.size()
        << ", waiting_queue_size:" << waiting_queue.size() << ", running_pending_reqs:" << running_pending_reqs.size()
        << ", swapped_queue_size:" << swapped_queue.size()
        << ", swapin_pending_requests_size:" << swapin_pending_requests_.size()
        << ", swapout_pending_requests_size:" << swapout_pending_requests_.size()
        << ", transfer_queue_size:" << transfer_queue.size() << ", step_sched_finish:" << step_sched_finish << ") ";
    return oss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const BatchState& b) {
    os << b.ToString();
    return os;
  }

  size_t multi_batch_id_;

  // The config of batch scheduler.
  BatchSchedulerConfig batch_scheduler_config_;

  // The waiting queue, double end queue.
  std::deque<std::shared_ptr<InferRequest>> waiting_queue;

  // The kv transfer queue, vector.
  std::vector<std::shared_ptr<InferRequest>> transfer_queue;

  // The buffer queue used to save finished swapin request temporary.
  std::vector<std::shared_ptr<InferRequest>> running_pending_reqs;

  std::vector<int64_t> merged_swapout_req_ids;
  std::vector<int64_t> merged_swapin_req_ids;

  // The swapped queue, sorted map.
  std::map<int, std::shared_ptr<InferRequest>> swapped_queue;

  // The pending requests used for swap in/out, unordered.
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapin_pending_requests_;
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapout_pending_requests_;

  // To guard queue.
  std::mutex queue_mutex;

  // The current timestamp for current schedule loop.
  uint64_t schedule_time_in_ms;

  // Whether current scheduler step have finished.
  bool step_sched_finish = false;

  // The current schedule output
  ScheduleOutput* schedule_output = nullptr;
};

}  // namespace ksana_llm
