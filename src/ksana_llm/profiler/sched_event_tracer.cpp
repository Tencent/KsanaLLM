/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/sched_event_tracer.h"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>

namespace ksana_llm {

int ScheduleEventTracer::trace_level_ = -1;

// Singleton implementation
ScheduleEventTracer* ScheduleEventTracer::GetInstance() {
  static ScheduleEventTracer instance;
  return &instance;
}

bool ScheduleEventTracer::IsEnabled() {
  if (trace_level_ < 0) {
    // Init trace_level_
    //  0:No trace, 1:batch trace, 2:req trace
    const char* trace_level_env = std::getenv("KLLM_SCHED_TRACE_LEVEL");
    trace_level_ = (trace_level_env ? std::stoi(trace_level_env) : 0);
    if ((trace_level_ < 0) || (trace_level_ > 2)) {
      trace_level_ = 0;
    }
  }
  return trace_level_ > 0;
}

ScheduleEventTracer::ScheduleEventTracer() : running_(true) {
  const char* node_rank_env = std::getenv("NODE_RANK");
  node_rank_ = node_rank_env ? std::stoi(node_rank_env) : 0;
  csv_file_path_ = "sched_events_node_" + std::to_string(node_rank_) + ".csv";

  // Start the worker thread
  worker_thread_ = std::thread(&ScheduleEventTracer::ProcessEvents, this);
}

ScheduleEventTracer::~ScheduleEventTracer() {
  // Signal the worker thread to stop
  running_ = false;

  // Wait for the worker thread to finish
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void ScheduleEventTracer::RecordRequestSchedEvent(RequestSchedEvent& event) {
  // Fast path: just add the event to the queue and return
  std::lock_guard<std::mutex> lock(queue_mutex_);
  event_queue_.push(event);
}

void ScheduleEventTracer::ProcessEvents() {
  std::ofstream csv_file(csv_file_path_);
  if (!csv_file.is_open()) {
    // Failed to open file
    return;
  }

  // Write CSV header
  csv_file << "event_type,node_rank,req_id,time_ns,phase,rank,pp_batch_idx,attn_dp_group_id,"
           << "forwarding_token_num,seq_len,req_num" << std::endl;

  std::queue<RequestSchedEvent> local_queue;

  while (running_ || !event_queue_.empty()) {
    // Sleep briefly to reduce CPU usage
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Get all events from the shared queue
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (!event_queue_.empty()) {
        // Swap the shared queue with our local queue for processing
        // This minimizes the time we hold the lock
        std::swap(local_queue, event_queue_);
      }
    }

    // Process all events in the local queue
    while (!local_queue.empty()) {
      const RequestSchedEvent& event = local_queue.front();

      if (trace_level_ > 0) {
        size_t forwarding_token_num = 0;
        size_t seq_len = 0;
        for (auto& req_info : event.batch_info.req_info_list) {
          forwarding_token_num += req_info.forwarding_token_num;
          seq_len += req_info.seq_len;
        }
        csv_file << event.event_type << "," << node_rank_ << ", -1 ," << event.time_ns << ","
                 << (event.phase == RequestEventPhase::Begin ? "Begin" : "End") << "," << event.rank << ","
                 << event.batch_info.pp_batch_idx << "," << event.attn_dp_group_id << "," << forwarding_token_num << ","
                 << seq_len << "," << event.batch_info.req_info_list.size() << std::endl;
      }
      if (trace_level_ > 1) {
        for (auto& req_info : event.batch_info.req_info_list) {
          // Convert event to CSV format and write to file
          csv_file << event.event_type << "," << node_rank_ << "," << req_info.req_id << "," << event.time_ns << ","
                   << (event.phase == RequestEventPhase::Begin ? "Begin" : "End") << "," << event.rank << ","
                   << event.batch_info.pp_batch_idx << "," << event.attn_dp_group_id << ","
                   << req_info.forwarding_token_num << "," << req_info.seq_len << "," << 1 << std::endl;
        }
      }
      local_queue.pop();
    }

    // Flush to ensure data is written to disk
    csv_file.flush();
  }

  csv_file.close();
}

}  // namespace ksana_llm
