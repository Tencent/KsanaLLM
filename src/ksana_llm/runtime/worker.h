/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Task structure for worker thread
struct WorkerTask {
  enum class TaskType { kForward, kSampling, kStop };

  TaskType type;
  std::promise<Status> promise;

  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  // Forward task parameters
  std::shared_ptr<BaseModel> model;
  std::shared_ptr<BaseWeight> weight;
  InferStage stage;
  std::vector<ForwardRequest>* forward_reqs;
  bool epilogue;
  RunMode run_mode;

  // Sampling task parameters
  std::shared_ptr<Sampler> sampler;
  std::vector<SamplingRequest>* sampling_reqs;
};

// The worker executed on every device.
class Worker {
 public:
  Worker(int rank, size_t pp_batch_num, std::shared_ptr<Context> context)
      : rank_(rank), pp_batch_num_(pp_batch_num), context_(context), running_(true) {
    // Create pp_batch_num worker threads
    worker_threads_.reserve(pp_batch_num_);
    for (size_t i = 0; i < pp_batch_num_; ++i) {
      worker_threads_.emplace_back(&Worker::ThreadLoop, this);
    }
  }

  ~Worker() {
    // Signal all threads to stop and wait for them
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      for (size_t i = 0; i < pp_batch_num_; ++i) {
        WorkerTask stop_task;
        stop_task.type = WorkerTask::TaskType::kStop;
        task_queue_.push(std::move(stop_task));
      }
      task_cv_.notify_all();  // Notify all threads
    }

    // Join all worker threads
    for (auto& thread : worker_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  // The async forward and sampling.
  std::future<Status> ForwardAsync(size_t multi_batch_id, std::shared_ptr<BaseModel> model,
                                   std::shared_ptr<BaseWeight> weight, InferStage stage,
                                   std::vector<ForwardRequest>& forward_reqs, bool epilogue,
                                   RunMode run_mode = RunMode::kMain);

  Status Forward(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                 InferStage stage, std::vector<ForwardRequest>& forward_reqs, bool epilogue,
                 RunMode run_mode = RunMode::kMain);

  std::future<Status> SamplingAsync(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                                    std::vector<SamplingRequest>& sampling_reqs);

  Status Sampling(size_t multi_batch_id, std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs);

 private:
  // Thread loop function that processes tasks
  void ThreadLoop();

  // Current worker rank.
  int rank_;

  // Number of parallel threads to use
  size_t pp_batch_num_;

  // GPU related context
  std::shared_ptr<Context> context_ = nullptr;

  // Worker threads
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;

  // Task queue
  std::queue<WorkerTask> task_queue_;
  std::mutex task_mutex_;
  std::condition_variable task_cv_;
};

// The worker group that used to manager multiple workers.
class WorkerGroup {
 public:
  WorkerGroup(size_t tensor_parallel_size, size_t pp_batch_num, std::shared_ptr<Context> context);
  ~WorkerGroup();

  // Get worker of specified rank.
  std::shared_ptr<Worker> GetWorker(int rank);

 private:
  // The inner workers.
  std::vector<std::shared_ptr<Worker>> workers_;

  // The parallel size.
  size_t tensor_parallel_size_;
};

}  // namespace ksana_llm
