/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <memory>

#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

MultiBatchController::MultiBatchController(int max_pp_batch_size) {
  task_threads_ready_flags_.resize(max_pp_batch_size, false);
}

void MultiBatchController::WaitUtilCurrentBatchCanRun(int multi_batch_id) {
  PROFILE_EVENT_SCOPE(WaitUtilCurrentBatchCanRun_, fmt::format("WaitUtilCurrentBatchCanRun_", multi_batch_id));
  std::unique_lock<std::mutex> lock(multi_batch_running_mtx_);
  KLLM_LOG_DEBUG << "wait to run multi_batch_id=" << multi_batch_id
                 << ", current_running_id=" << current_running_multi_batch_id_;
  if (current_running_multi_batch_id_ == kInvalidMultiBatchId) {
    constexpr bool force_change = true;
    current_running_multi_batch_id_ = GetNextRunningBatchId(current_running_multi_batch_id_, force_change);
  }
  multi_batch_keep_order_cv_.wait(
      lock, [this, multi_batch_id]() { return multi_batch_id == current_running_multi_batch_id_; });
  KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " running.";
}

void MultiBatchController::NotifyOtherBatchCanRun(bool force_change) {
  std::unique_lock<std::mutex> lock(multi_batch_running_mtx_);
  auto last_id = current_running_multi_batch_id_;
  current_running_multi_batch_id_ = GetNextRunningBatchId(current_running_multi_batch_id_, force_change);
  KLLM_LOG_DEBUG << "unlocked current multi_batch_id=" << last_id
                 << ", and multi_batch_id=" << current_running_multi_batch_id_ << " can be run.";
  multi_batch_keep_order_cv_.notify_all();
}

int MultiBatchController::GetNextRunningBatchId(int cur_multi_batch_id, bool force_change) {
  // using under multi_batch_running_mtx_ scope, do not need lock again
  int current_id = cur_multi_batch_id;
  current_id++;
  int total_cnt = task_threads_ready_flags_.size();
  for (size_t i = 0; i < total_cnt; ++i) {
    if (current_id >= total_cnt) {
      current_id = 0;
    }
    if (task_threads_ready_flags_.at(current_id)) {
      return current_id;
    }
    current_id++;
  }
  if (force_change) {
    KLLM_LOG_WARNING << "No more multi_batch task threads ready.";
    return kInvalidMultiBatchId;
  } else {
    KLLM_LOG_DEBUG << "all other batch threads are not ready, next keep using current batch id " << cur_multi_batch_id;
    return cur_multi_batch_id;
  }
}

int MultiBatchController::GetRunningTasksNum() {
  std::unique_lock<std::mutex> lock(multi_batch_running_mtx_);
  int cnt = 0;
  for (size_t i = 0; i < task_threads_ready_flags_.size(); ++i) {
    if (task_threads_ready_flags_[i]) {
      cnt++;
    }
  }
  return cnt;
}

void MultiBatchController::WaitUtilCanRecvCurrentHiddenUnits(int multi_batch_id) {
  PROFILE_EVENT_SCOPE(WaitUtilCanRecv_, fmt::format("WaitUtilCanRecv_", multi_batch_id));
  KLLM_LOG_DEBUG << "start waiting to recv multi_batch_id=" << multi_batch_id
                 << ", current running:" << current_running_multi_batch_id_;
  std::unique_lock<std::mutex> lock(multi_batch_need_recv_mtx_);
  multi_batch_can_recv_hiddens_cv_.wait(lock, [this, multi_batch_id] {
    if (need_recv_hiddens_multi_batch_ids_.size() > 1) {
      int last_id = need_recv_hiddens_multi_batch_ids_.front();
      return last_id == multi_batch_id;
    } else {
      return GetRunningTasksNum() <= 1;
    }
  });
  need_recv_hiddens_multi_batch_ids_.pop();
  KLLM_LOG_DEBUG << "now can recv multi_batch_id=" << multi_batch_id
                 << ", current running_id:" << current_running_multi_batch_id_;
}

void MultiBatchController::NotifyLastBatchHiddenUnitCanRecv(int cur_muilt_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_need_recv_mtx_);
  need_recv_hiddens_multi_batch_ids_.push(cur_muilt_batch_id);
  multi_batch_can_recv_hiddens_cv_.notify_all();
  KLLM_LOG_DEBUG << "set can recv last multi_batch_id=" << need_recv_hiddens_multi_batch_ids_.front();
}

void MultiBatchController::MultiBatchThreadIsReady(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_running_mtx_);
  task_threads_ready_flags_.at(multi_batch_id) = true;
}

void MultiBatchController::MultiBatchThreadIsNotReady(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_running_mtx_);
  task_threads_ready_flags_.at(multi_batch_id) = false;
}

void MultiBatchController::NotifyCurrentBatchThreadNotReady(int cur_muilt_batch_id) {
  KLLM_LOG_DEBUG << "notify not ready cur_muilt_batch_id=" << cur_muilt_batch_id;
  MultiBatchThreadIsNotReady(cur_muilt_batch_id);
  constexpr bool force_change = true;
  NotifyOtherBatchCanRun(force_change);
  multi_batch_can_recv_hiddens_cv_.notify_all();
}

}  // namespace ksana_llm
