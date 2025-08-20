/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>

namespace ksana_llm {

// Status of batch:
//   NotReady: batch not ready to run, next status will be Standby
//   Standby: batch is ready to run, next status will be Running
//   Running: batch is running, next status will be Finished
//   Finished: batch is finished, next status will be NotReady
enum BatchStatus {
  NotReady = 0,
  Standby = 1,
  Running = 2,
  Finished = 3,
};

class MultiBatchController  {
 public:
  static constexpr int kEmptyMultiBatchId = -1;

  explicit MultiBatchController(int max_batch_size);

  // mark current multi_batch_id is ready to run
  void NotifyCurrentBatchIsStandby(int multi_batch_id);

  ////// for running order
  // wait util current multi_batch_id can be running
  void WaitUntilCurrentBatchCanRun(int multi_batch_id);

  // Notify current multi_batch_id can be running, and keep current multi_batch_id or not
  void NotifyAnotherBatchCanRun(int multi_batch_id);

  // Wait util current multi_batch_id can recv hidden units
  void WaitUtilCanRecvCurrentHiddenUnits(int cur_multi_batch_id);

  ////// for recv hiddens order
  // Notify last batch id can recv hidden units at current id
  void NotifyLastBatchHiddenUnitCanRecv(int cur_muilt_batch_id);

  // Notify current batch id not ready
  void NotifyCurrentBatchThreadNotReady(int multi_batch_id);

  // Notify current batch id is finish
  void NotifyCurrentBatchIsFinish(int multi_batch_id);

 private:
  // batch id inc 1, and return next id which can run
  // return -1 if none can run
  int GetNextRunningBatchId(int cur_multi_batch_id);

  // called in NotifyCurrentBatchThreadNotReady, only can change
  void WaitUntilCurrentBatchIsFinish(int multi_batch_id);

  // called in NotifyCurrentBatchThreadNotReady
  void SetCurrentBatchIsNotReady(int multi_batch_id);

 private:
  // Note(TJ): use int type to return -1
  int next_running_multi_batch_id_ = kEmptyMultiBatchId;

  // for running order
  std::mutex multi_batch_status_mtx_;
  std::condition_variable multi_batch_status_cv_;
  std::vector<BatchStatus> tasks_status_;

  // for recv hiddens order
  std::queue<int> need_recv_hiddens_multi_batch_ids_;
};

}  // namespace ksana_llm
