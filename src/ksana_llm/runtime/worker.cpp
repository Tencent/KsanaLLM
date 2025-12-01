/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/worker.h"
#ifdef ENABLE_CUDA
#  include <c10/cuda/CUDAFunctions.h>
#endif

#include <pthread.h>
#include <memory>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status Worker::Forward(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                       std::vector<ForwardRequest*>& forward_reqs, bool epilogue, RunMode run_mode) {
  return model->Forward(multi_batch_id, weight, forward_reqs, epilogue, run_mode);
}

// This function is designed to be run by multiple threads in parallel
void Worker::ThreadLoop() {
  // 设置线程名称用于调试和监控
  constexpr size_t kMaxThreadNameLength = 15;
  std::string thread_name = "worker_" + std::to_string(rank_);
#ifdef __linux__
  // 在Linux上设置线程名称 (最大15个字符 + 空终止符)
  if (thread_name.length() > kMaxThreadNameLength) {
    thread_name = thread_name.substr(0, kMaxThreadNameLength);
  }
  pthread_setname_np(pthread_self(), thread_name.c_str());
#endif
  KLLM_LOG_INFO << "thread_name " << thread_name;
  SetDevice(rank_);

  while (running_.load(std::memory_order_acquire)) {
    WorkerTask task;
    if (!GetTask(task)) {
      continue;
    }

    PROFILE_EVENT_SCOPE(task_thread_, fmt::format("task_thread_{}", task.multi_batch_id));

    switch (task.type) {
      case WorkerTask::TaskType::kForward: {
        Forward(task.multi_batch_id, task.model, task.weight, *task.forward_reqs, task.epilogue, task.run_mode);
        break;
      }
      case WorkerTask::TaskType::kSampling: {
        Sampling(task.multi_batch_id, task.sampler, *task.sampling_reqs);
        break;
      }
      case WorkerTask::TaskType::kStop: {
        running_.store(false, std::memory_order_release);
        break;
      }
    }
    if (task.wg) {
      task.wg->Done();
    }
  }
}

void Worker::ForwardAsync(size_t multi_batch_id, std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                          std::vector<ForwardRequest*>& forward_reqs, bool epilogue, std::shared_ptr<WaitGroup> wg,
                          RunMode run_mode) {
  PROFILE_EVENT_SCOPE(ForwardAsync, fmt::format("ForwardAsync_{}", multi_batch_id));
  WorkerTask task;
  task.type = WorkerTask::TaskType::kForward;
  task.wg = wg;
  task.multi_batch_id = multi_batch_id;
  task.model = model;
  task.weight = weight;
  task.forward_reqs = &forward_reqs;
  task.epilogue = epilogue;
  task.run_mode = run_mode;
  AddTask(std::move(task));
}

Status Worker::Sampling(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                        std::vector<SamplingRequest*>& sampling_reqs) {
  PROFILE_EVENT_SCOPE(task_sampling_, fmt::format("task_sampling_{}", multi_batch_id));
  return sampler->Sampling(multi_batch_id, sampling_reqs, context_->GetComputeStreams()[rank_]);
}

void Worker::SamplingAsync(size_t multi_batch_id, std::shared_ptr<Sampler> sampler,
                           std::vector<SamplingRequest*>& sampling_reqs, std::shared_ptr<WaitGroup> wg) {
  WorkerTask task;
  task.multi_batch_id = multi_batch_id;
  task.type = WorkerTask::TaskType::kSampling;
  task.wg = wg;
  task.sampler = sampler;
  task.sampling_reqs = &sampling_reqs;
  AddTask(std::move(task));
}

WorkerGroup::WorkerGroup(const size_t tensor_parallel_size, const size_t pp_batch_num,
                         std::shared_ptr<Context> context) {
#ifdef ENABLE_CUDA
  // Used to force libtorch cudaStream initialized.
  for (size_t dev_id = 0; dev_id < tensor_parallel_size; ++dev_id) {
    SetDevice(dev_id);
    c10::cuda::set_device(dev_id);
    const auto int32_options = torch::TensorOptions().device(torch::kCUDA, dev_id).dtype(torch::kInt32);
    const torch::Tensor tensor = torch::ones({1024, 1}, int32_options);
  }
#endif
  workers_.resize(tensor_parallel_size);
  for (size_t worker_id = 0; worker_id < tensor_parallel_size; ++worker_id) {
    workers_[worker_id].reset(new Worker(worker_id, pp_batch_num, context));
  }
}

}  // namespace ksana_llm
