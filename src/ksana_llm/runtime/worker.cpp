/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/worker.h"

#include <memory>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status Worker::Forward(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                       InferStage stage, std::vector<ForwardRequest>& forward_reqs, bool epilogue, RunMode run_mode) {
  // TODO(karlluo): confirm redundant usage
  SetDevice(rank_);
  opentelemetry::trace::StartSpanOptions options;

  KLLM_LOG_DEBUG << "forwarding_tokens_num " << forward_reqs[0].forwarding_tokens[0].size() << ", req_id "
                << forward_reqs[0].req_id << ",kv_cached_token_num " << forward_reqs[0].kv_cached_token_num
                << ", prefix_cache_len" << forward_reqs[0].prefix_cache_len << ", tokens[1]_num "
                << forward_reqs[0].forwarding_tokens[1].size();
  return model->Forward(multi_batch_id, weight, forward_reqs, epilogue, run_mode);
}

// This function is designed to be run by multiple threads in parallel
void Worker::ThreadLoop() {
  while (running_) {
    WorkerTask task;

    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cv_.wait(lock, [this] { return !task_queue_.empty(); });

      task = std::move(task_queue_.front());
      task_queue_.pop();
    }
    PROFILE_EVENT_SCOPE(task_thread_, fmt::format("task_thread_{}", task.multi_batch_id));

    switch (task.type) {
      case WorkerTask::TaskType::kForward: {
        Status status = Forward(task.multi_batch_id, task.model, task.weight, task.stage, *task.forward_reqs,
                                task.epilogue, task.run_mode);
        task.promise.set_value(status);
        break;
      }
      case WorkerTask::TaskType::kSampling: {
        Status status = Sampling(task.multi_batch_id, task.sampler, *task.sampling_reqs);
        task.promise.set_value(status);
        break;
      }
      case WorkerTask::TaskType::kStop: {
        running_ = false;
        break;
      }
    }
  }
}

std::future<Status> Worker::ForwardAsync(size_t multi_batch_id, std::shared_ptr<BaseModel> model,
                                         std::shared_ptr<BaseWeight> weight, InferStage stage,
                                         std::vector<ForwardRequest>& forward_reqs, bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(ForwardAsync, fmt::format("ForwardAsync_{}", multi_batch_id));
  std::promise<Status> promise;
  std::future<Status> future = promise.get_future();

  WorkerTask task;
  task.type = WorkerTask::TaskType::kForward;
  task.promise = std::move(promise);
  task.multi_batch_id = multi_batch_id;
  task.model = model;
  task.weight = weight;
  task.stage = stage;
  task.forward_reqs = &forward_reqs;
  task.epilogue = epilogue;
  task.run_mode = run_mode;

  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    task_queue_.emplace(std::move(task));
    task_cv_.notify_one();  // Notify one thread is sufficient as we only need one thread to handle this task
  }

  return future;
}

Status Worker::Sampling(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                        std::vector<SamplingRequest>& sampling_reqs) {
  PROFILE_EVENT_SCOPE(task_sampling_, fmt::format("task_sampling_{}", multi_batch_id));
  // TODO(karlluo): confirm redundant usage
  SetDevice(rank_);
  return sampler->Sampling(multi_batch_id, sampling_reqs, context_->GetComputeStreams()[rank_]);
}

std::future<Status> Worker::SamplingAsync(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                                          std::vector<SamplingRequest>& sampling_reqs) {
  std::promise<Status> promise;
  std::future<Status> future = promise.get_future();

  WorkerTask task;
  task.multi_batch_id = multi_batch_id;
  task.type = WorkerTask::TaskType::kSampling;
  task.promise = std::move(promise);
  task.sampler = sampler;
  task.sampling_reqs = &sampling_reqs;

  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    task_queue_.push(std::move(task));
    task_cv_.notify_one();  // Notify one thread is sufficient as we only need one thread to handle this task
  }

  return future;
}

WorkerGroup::WorkerGroup(size_t tensor_parallel_size, size_t pp_batch_num, std::shared_ptr<Context> context)
    : tensor_parallel_size_(tensor_parallel_size) {
  workers_.resize(tensor_parallel_size_);
  for (size_t worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    workers_[worker_id].reset(new Worker(worker_id, pp_batch_num, context));
  }
}

WorkerGroup::~WorkerGroup() {
  // Worker destructor will handle thread cleanup
}

std::shared_ptr<Worker> WorkerGroup::GetWorker(int rank) {
  if (rank < 0 || rank >= static_cast<int>(workers_.size())) {
    KLLM_LOG_FATAL << "The worker rank " << rank << " exceed worker size " << workers_.size();
  }
  return workers_[rank];
}

}  // namespace ksana_llm
