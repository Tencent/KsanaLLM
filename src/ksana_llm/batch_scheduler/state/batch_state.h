/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/request_state.h"

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

  void ResetInfoBeforeSchedule() {
    schedule_time_in_ms = GetCurrentTimeInMs();
    step_sched_finish = false;

    schedule_output->multi_batch_id += 1;
  }

  std::string ToString(bool include_details = false) const {
    std::ostringstream oss;
    oss << " BatchState(multi_batch_id:" << multi_batch_id_ << ", waiting_queue_size:" << waiting_queue.size()
        << ", decoding_queue_size:" << decoding_queue.size();
    if (transfer_queue.size() > 0) {
      oss << ", transfer_queue_size:" << transfer_queue.size();
    }
    if (async_stoped_reqs.size() > 0) {
      oss << ", async_stop_req_size:" << async_stoped_reqs.size();
    }
    if (async_recomputed_reqs.size() > 0) {
      oss << ", async_recompute_req_size:" << async_recomputed_reqs.size();
    }
    if (async_swapout_reqs.size() > 0) {
      oss << ", async_swapout_reqs_size:" << async_swapout_reqs.size();
    }
    if (swapped_queue.size() > 0) {
      oss << ", swapped_queue_size:" << swapped_queue.size();
    }
    if (swapin_pending_requests.size() > 0) {
      oss << ", swapin_pending_requests_size:" << swapin_pending_requests.size();
    }
    if (swapout_pending_requests.size() > 0) {
      oss << ", swapout_pending_requests_size:" << swapout_pending_requests.size();
    }

    oss << ", step_sched_finish:" << step_sched_finish << ") ";

    if (include_details) {
      oss << ", waiting_queue:[";
      for (const auto& req : waiting_queue) {
        oss << req->req_id << " ";
      }
      oss << "], decoding_queue:[";
      for (const auto& req : decoding_queue) {
        oss << req->req_id << " ";
      }
      oss << "]";
      if (transfer_queue.size() > 0) {
        oss << ", transfer_queue:[";
        for (const auto& req : transfer_queue) {
          oss << req->req_id << " ";
        }
        oss << "]";
      }
      if (swapped_queue.size() > 0) {
        oss << ", swapped_queue:[";
        for (const auto& req : swapped_queue) {
          oss << req.second->req_id << " ";
        }
        oss << "]";
      }

      if (async_recomputed_reqs.size() > 0) {
        oss << ", async_recomputed_reqs:[";
        for (const auto& req : async_recomputed_reqs) {
          oss << req.second->req_id << " ";
        }
        oss << "]";
      }
      if (async_swapout_reqs.size() > 0) {
        oss << ", async_swapout_reqs:[";
        for (const auto& req : async_swapout_reqs) {
          oss << req.second->req_id << " ";
        }
        oss << "]";
      }
    }
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
  std::vector<std::shared_ptr<InferRequest>> waiting_queue;

  // The mocked req queue， used to make sure attn has one req at least.
  std::vector<std::shared_ptr<InferRequest>> mock_queue;

  // The kv transfer queue, vector.
  std::vector<std::shared_ptr<InferRequest>> transfer_queue;

  // Running decoding requests
  std::vector<std::shared_ptr<InferRequest>> decoding_queue;

  struct StoppedReqInfo {
    std::shared_ptr<InferRequest> infer_request;
    Status ret_status;
    RequestState req_state;
  };
  std::unordered_map<int64_t, StoppedReqInfo> async_stoped_reqs;
  std::unordered_map<int64_t, std::shared_ptr<InferRequest>> async_recomputed_reqs;
  std::unordered_map<int64_t, std::shared_ptr<InferRequest>> async_swapout_reqs;

  std::vector<int64_t> merged_swapout_req_ids;
  std::vector<int64_t> merged_swapin_req_ids;

  // The swapped queue, sorted map.
  std::map<int, std::shared_ptr<InferRequest>> swapped_queue;

  // The pending requests used for swap in/out, unordered.
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapin_pending_requests;
  std::unordered_map<int, std::shared_ptr<InferRequest>> swapout_pending_requests;

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
