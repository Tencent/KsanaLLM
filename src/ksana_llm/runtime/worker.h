/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// Task structure for worker thread
struct WorkerTask {
  enum class TaskType { kForward, kSampling, kStop };
  TaskType type;
  std::shared_ptr<WaitGroup> wg;
  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  // Forward task parameters
  std::shared_ptr<BaseModel> model;
  std::shared_ptr<BaseWeight> weight;
  std::vector<ForwardRequest*>* forward_reqs;
  bool epilogue;
  RunMode run_mode;

  // Sampling task parameters
  std::shared_ptr<Sampler> sampler;
  std::vector<SamplingRequest>* sampling_reqs;
};

// The worker executed on every device.
class Worker {
 public:
  Worker(const int rank, const size_t pp_batch_num, std::shared_ptr<Context> context)
      : rank_(rank), pp_batch_num_(pp_batch_num), context_(context), running_(true) {
    // Create pp_batch_num worker threads
    worker_threads_.reserve(pp_batch_num_);
    for (size_t i = 0; i < pp_batch_num_; ++i) {
      worker_threads_.emplace_back(&Worker::ThreadLoop, this);
    }
  }

  ~Worker() {
    // Signal all threads to stop and wait for them
    for (size_t i = 0; i < pp_batch_num_; ++i) {
      WorkerTask stop_task;
      stop_task.type = WorkerTask::TaskType::kStop;
      AddTask(std::move(stop_task));
    }

    // Join all worker threads
    for (auto& thread : worker_threads_) {
      thread.join();
    }
  }

  void AddTask(WorkerTask&& task) {
    while (queue_lock_.test_and_set(std::memory_order_acquire)) {
      __builtin_ia32_pause();
    }
    task_queue_.emplace(std::move(task));
    queue_lock_.clear(std::memory_order_release);
  }

  bool GetTask(WorkerTask& task) {
    while (queue_lock_.test_and_set(std::memory_order_acquire)) {
      __builtin_ia32_pause();
    }

    const bool has_task = !task_queue_.empty();
    if (has_task) {
      task = std::move(task_queue_.front());
      task_queue_.pop();
    }

    queue_lock_.clear(std::memory_order_release);
    return has_task;
  }

  void ForwardAsync(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                    std::vector<ForwardRequest*>& forward_reqs, bool epilogue, std::shared_ptr<WaitGroup> wg,
                    RunMode run_mode = RunMode::kMain);

  Status Forward(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                 std::vector<ForwardRequest*>& forward_reqs, bool epilogue, RunMode run_mode = RunMode::kMain);

  void SamplingAsync(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                     std::vector<SamplingRequest>& sampling_reqs, std::shared_ptr<WaitGroup> wg);

  Status Sampling(size_t multi_batch_id, std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs);

 private:
  // Thread loop function that processes tasks
  void ThreadLoop();

  // Current worker rank.
  const int rank_;

  // Number of parallel threads to use
  const size_t pp_batch_num_;

  // GPU related context
  std::shared_ptr<Context> context_ = nullptr;

  // Worker threads
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;

  // Task queue with spinlock for high-performance consumption
  std::queue<WorkerTask> task_queue_;
  std::atomic_flag queue_lock_ = ATOMIC_FLAG_INIT;
};

// The worker group that used to manager multiple workers.
class WorkerGroup {
 public:
  WorkerGroup(const size_t tensor_parallel_size, const size_t pp_batch_num, std::shared_ptr<Context> context);

  // Get worker of specified rank.
  std::shared_ptr<Worker> GetWorker(const int rank) const { return workers_[rank]; }

 private:
  // The inner workers.
  std::vector<std::shared_ptr<Worker>> workers_;
};

}  // namespace ksana_llm
