/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>

namespace ksana_llm {

class MultiBatchController  {
 public:
  explicit MultiBatchController(int max_pp_batch_size);

  // mark current multi_batch_id is ready
  void MultiBatchThreadIsReady(int multi_batch_id);

  ////// for running order
  // wait util current multi_batch_id can be running
  void WaitUtilCurrentBatchCanRun(int multi_batch_id);

  // Notify current multi_batch_id can be running, and keep current multi_batch_id or not
  void NotifyOtherBatchCanRun(bool force_change = false);

  // Wait util current multi_batch_id can recv hidden units
  void WaitUtilCanRecvCurrentHiddenUnits(int multi_batch_id);

  ////// for recv hiddens order
  // Notify last batch id can recv hidden units at current id
  void NotifyLastBatchHiddenUnitCanRecv(int cur_muilt_batch_id);

  // Notify current batch id not ready
  void NotifyCurrentBatchThreadNotReady(int multi_batch_id);

  // get running task threads num
  int GetRunningTasksNum();

 private:
  // called in NotifyCurrentBatchThreadNotReady
  void MultiBatchThreadIsNotReady(int multi_batch_id);

  // Get next running multi_batch_id, if no threads are ready and force_change is set will return -1
  int GetNextRunningBatchId(int multi_batch_id, bool force_change);

 private:
  static constexpr int kInvalidMultiBatchId = -1;
  // use int type to return -1
  int current_running_multi_batch_id_ = 0;

  // for running order
  std::mutex multi_batch_running_mtx_;
  std::condition_variable multi_batch_keep_order_cv_;
  std::vector<bool> task_threads_ready_flags_;

  // for recv hiddens order
  std::mutex multi_batch_need_recv_mtx_;
  std::queue<int> need_recv_hiddens_multi_batch_ids_;
  std::condition_variable multi_batch_can_recv_hiddens_cv_;
};

}  // namespace ksana_llm
