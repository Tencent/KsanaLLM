/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "timer.h"

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

// Define event start and end point
enum RequestEventPhase { Begin, End };

struct BatchRequestSchedInfo {
  struct RequestSchedInfo {
    int req_id;
    size_t forwarding_token_num;
    size_t seq_len;
  };
  size_t multi_batch_id;
  std::vector<RequestSchedInfo> req_info_list;
};

struct RequestSchedEvent {
  const char* event_type;
  std::time_t time_ns;
  RequestEventPhase phase = RequestEventPhase::Begin;
  int rank;
  size_t attn_dp_group_id;  // data parallel id of this request.
  BatchRequestSchedInfo batch_info;
};

class ScheduleEventTracer {
 public:
  ScheduleEventTracer();
  ~ScheduleEventTracer();

  static ScheduleEventTracer* GetInstance();
  static bool IsEnabled();

  // Fast method to record an event - just adds to queue and returns
  void RecordRequestSchedEvent(RequestSchedEvent& event);

 private:
  // Process events from the queue and write to CSV
  void ProcessEvents();

  std::atomic<bool> running_;
  std::mutex queue_mutex_;
  std::thread worker_thread_;
  std::string csv_file_path_;
  int node_rank_;
  std::queue<RequestSchedEvent> event_queue_;

  static int trace_level_;
};

inline BatchRequestSchedInfo BuildBatchRequestSchedInfoFromInferReqs(
    const std::vector<std::shared_ptr<InferRequest>>& reqs, size_t multi_batch_id) {
  BatchRequestSchedInfo batch_info;
  batch_info.multi_batch_id = multi_batch_id;
  batch_info.req_info_list.resize(reqs.size());
  for (size_t i = 0; i < reqs.size(); i++) {
    auto& req = reqs[i];
    auto& req_info = batch_info.req_info_list[i];
    req_info.req_id = req->req_id;
    req_info.forwarding_token_num = req->forwarding_tokens.size() - req->kv_cached_token_num;
    req_info.seq_len = req->forwarding_tokens.size();
  }
  return batch_info;
}

inline BatchRequestSchedInfo BuildBatchRequestSchedInfoFromForwardingReqs(const std::vector<ForwardRequest>& reqs,
                                                                          size_t multi_batch_id) {
  BatchRequestSchedInfo batch_info;
  batch_info.multi_batch_id = multi_batch_id;
  batch_info.req_info_list.resize(reqs.size());
  for (size_t i = 0; i < reqs.size(); i++) {
    auto& req = reqs[i];
    auto& req_info = batch_info.req_info_list[i];
    req_info.req_id = req.req_id;
    req_info.forwarding_token_num = req.forwarding_tokens->size() - req.kv_cached_token_num;
    req_info.seq_len = req.forwarding_tokens->size();
  }
  return batch_info;
}

inline void RecordRequestSchedEventsWithTime(BatchRequestSchedInfo& batch_sched_info, int rank, size_t attn_dp_group_id,
                                             const char* type, RequestEventPhase phase, time_t time_ns) {
  if (!ScheduleEventTracer::IsEnabled()) return;
  if (rank != 0) return;  // only record first card. TODO(karlluo): record rank 0 of different dp group
  if (batch_sched_info.req_info_list.size() == 0) return;
  RequestSchedEvent event;
  event.batch_info = batch_sched_info;
  event.event_type = type;
  event.time_ns = time_ns;
  event.phase = phase;
  event.rank = rank;
  event.attn_dp_group_id = attn_dp_group_id;
  event.batch_info = batch_sched_info;
  ScheduleEventTracer::GetInstance()->RecordRequestSchedEvent(event);
}

inline void RecordRequestSchedEvents(BatchRequestSchedInfo& batch_sched_info, int rank, size_t attn_dp_group_id,
                                     const char* type, RequestEventPhase phase) {
  RecordRequestSchedEventsWithTime(batch_sched_info, rank, attn_dp_group_id, type, phase,
                                   ProfileTimer::GetCurrentTimeInNs());
}

inline void RecordRequestSchedEvents(const std::vector<std::shared_ptr<InferRequest>>& reqs, int rank,
                                     size_t multi_batch_id, size_t attn_dp_group_id, const char* type,
                                     RequestEventPhase phase) {
  time_t time_ns = ProfileTimer::GetCurrentTimeInNs();
  auto batch_info = BuildBatchRequestSchedInfoFromInferReqs(reqs, multi_batch_id);
  RecordRequestSchedEventsWithTime(batch_info, rank, attn_dp_group_id, type, phase, time_ns);
}

inline void RecordRequestSchedEventsWithStartTime(std::vector<std::shared_ptr<InferRequest>> reqs, int rank,
                                                  size_t multi_batch_id, size_t attn_dp_group_id, const char* type,
                                                  time_t start_time_ns) {
  auto batch_info = BuildBatchRequestSchedInfoFromInferReqs(reqs, multi_batch_id);
  RecordRequestSchedEventsWithTime(batch_info, rank, attn_dp_group_id, type, RequestEventPhase::Begin, start_time_ns);
  time_t time_ns = ProfileTimer::GetCurrentTimeInNs();
  RecordRequestSchedEventsWithTime(batch_info, rank, attn_dp_group_id, type, RequestEventPhase::End, time_ns);
}

}  // namespace ksana_llm
