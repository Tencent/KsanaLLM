/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <memory>

#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

MultiBatchController::MultiBatchController(int max_batch_size) {
  tasks_status_.resize(max_batch_size);
  for (size_t i = 0; i < tasks_status_.size(); ++i) {
    tasks_status_[i] = BatchStatus::NotReady;
  }
}

void MultiBatchController::WaitUntilCurrentBatchCanRun(int cur_multi_batch_id) {
  PROFILE_EVENT_SCOPE(WaitUntilCurrentBatchCanRun_, fmt::format("WaitUntilCurrentBatchCanRun_", cur_multi_batch_id));
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  KLLM_LOG_SCHEDULER << "wait to run multi_batch_id=" << cur_multi_batch_id;
  multi_batch_status_cv_.wait(lock, [this, cur_multi_batch_id]() {
    if (tasks_status_.at(cur_multi_batch_id) == BatchStatus::Running) {
      KLLM_LOG_SCHEDULER << "can run multi_batch_id=" << cur_multi_batch_id << " since this batch is already running.";
      return true;
    }
    for (size_t i = 0; i < tasks_status_.size(); ++i) {
      if (tasks_status_.at(i) == BatchStatus::Running) {
        // some other batch is still running, this batch can not run
        return false;
      }
    }
    if (tasks_status_.at(cur_multi_batch_id) != BatchStatus::Standby) {
      return false;
    }
    if (cur_multi_batch_id == next_running_multi_batch_id_ || next_running_multi_batch_id_ == kEmptyMultiBatchId) {
      tasks_status_.at(cur_multi_batch_id) = BatchStatus::Running;
      next_running_multi_batch_id_ = cur_multi_batch_id;
      return true;
    } else {
      return false;
    }
  });
  KLLM_LOG_SCHEDULER << "multi_batch_id=" << cur_multi_batch_id << " is running.";
}

int MultiBatchController::GetNextRunningBatchId(int cur_multi_batch_id) {
  // using under multi_batch_running_mtx_ scope, do not need lock again
  int current_id = cur_multi_batch_id;
  current_id++;
  int total_cnt = tasks_status_.size();
  for (size_t i = 0; i < total_cnt; ++i) {
    if (current_id >= total_cnt) {
      current_id = 0;
    }
    if (tasks_status_.at(current_id) == BatchStatus::Standby) {
      return current_id;
    }
    current_id++;
  }
  return kEmptyMultiBatchId;
}

void MultiBatchController::NotifyAnotherBatchCanRun(int cur_multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  bool other_task_has_Standby = false;
  for (size_t i = 0; i < tasks_status_.size(); ++i) {
    if (i != cur_multi_batch_id && tasks_status_.at(i) == BatchStatus::Standby) {
      other_task_has_Standby = true;
      break;
    }
  }
  if (other_task_has_Standby && tasks_status_.at(cur_multi_batch_id) == BatchStatus::Running) {
    next_running_multi_batch_id_ = GetNextRunningBatchId(cur_multi_batch_id);
    tasks_status_.at(cur_multi_batch_id) = BatchStatus::Standby;
    KLLM_LOG_SCHEDULER << "stop runing multi_batch_id=" << cur_multi_batch_id
                       << ", and notify multi_batch_id=" << next_running_multi_batch_id_ << " can run";
  } else {
    KLLM_LOG_SCHEDULER << "no other batches ready, keep nothing changed multi_batch_id=" << cur_multi_batch_id;
  }
  multi_batch_status_cv_.notify_all();
}

void MultiBatchController::WaitUtilCanRecvCurrentHiddenUnits(int cur_multi_batch_id) {
  PROFILE_EVENT_SCOPE(WaitUtilCanRecv_, fmt::format("WaitUtilCanRecvCurrentHiddenUnits", cur_multi_batch_id));
  KLLM_LOG_SCHEDULER << "start waiting to recv multi_batch_id=" << cur_multi_batch_id;
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  multi_batch_status_cv_.wait(lock, [this, cur_multi_batch_id] {
    if (tasks_status_.at(cur_multi_batch_id) == BatchStatus::Running) {
      KLLM_LOG_SCHEDULER << "can recv multi_batch_id=" << cur_multi_batch_id << " since this batch is already running.";
      return true;
    }
    if (need_recv_hiddens_multi_batch_ids_.size() > 1) {
      int last_id = need_recv_hiddens_multi_batch_ids_.front();
      return last_id == cur_multi_batch_id;
    } else {
      int ready_cnt = 0;
      for (size_t i = 0; i < tasks_status_.size(); ++i) {
        if (tasks_status_.at(i) == BatchStatus::Standby || tasks_status_.at(i) == BatchStatus::Running) {
          ready_cnt++;
        }
      }
      return ready_cnt <= 1;
    }
  });
  need_recv_hiddens_multi_batch_ids_.pop();
  KLLM_LOG_SCHEDULER << "now can recv multi_batch_id=" << cur_multi_batch_id;
}

void MultiBatchController::NotifyLastBatchHiddenUnitCanRecv(int cur_muilt_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  need_recv_hiddens_multi_batch_ids_.push(cur_muilt_batch_id);
  multi_batch_status_cv_.notify_all();
  KLLM_LOG_SCHEDULER << "set can recv last multi_batch_id=" << need_recv_hiddens_multi_batch_ids_.front();
}

void MultiBatchController::NotifyCurrentBatchIsStandby(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  tasks_status_.at(multi_batch_id) = BatchStatus::Standby;
  multi_batch_status_cv_.notify_all();
  KLLM_LOG_SCHEDULER << "notify muilt_batch_id=" << multi_batch_id << " is ready";
}

void MultiBatchController::SetCurrentBatchIsNotReady(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  tasks_status_.at(multi_batch_id) = BatchStatus::NotReady;
  if (next_running_multi_batch_id_ == multi_batch_id) {
    next_running_multi_batch_id_ = GetNextRunningBatchId(multi_batch_id);
  }
  KLLM_LOG_SCHEDULER << "notify muilt_batch_id=" << multi_batch_id << " is not ready";
}

void MultiBatchController::NotifyCurrentBatchIsFinish(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  tasks_status_.at(multi_batch_id) = BatchStatus::Finished;
  multi_batch_status_cv_.notify_all();
  KLLM_LOG_SCHEDULER << "notify muilt_batch_id=" << multi_batch_id << " is finished";
}

void MultiBatchController::WaitUntilCurrentBatchIsFinish(int multi_batch_id) {
  std::unique_lock<std::mutex> lock(multi_batch_status_mtx_);
  KLLM_LOG_SCHEDULER << "wait util finish multi_batch_id=" << multi_batch_id;
  multi_batch_status_cv_.wait(lock, [this, multi_batch_id]() {
    return tasks_status_.at(multi_batch_id) == BatchStatus::Finished ||
           tasks_status_.at(multi_batch_id) == BatchStatus::NotReady;
  });
}

void MultiBatchController::NotifyCurrentBatchThreadNotReady(int cur_muilt_batch_id) {
  KLLM_LOG_SCHEDULER << "notify not ready cur_muilt_batch_id=" << cur_muilt_batch_id;
  WaitUntilCurrentBatchIsFinish(cur_muilt_batch_id);
  NotifyAnotherBatchCanRun(cur_muilt_batch_id);
  SetCurrentBatchIsNotReady(cur_muilt_batch_id);
  multi_batch_status_cv_.notify_all();
}

}  // namespace ksana_llm
