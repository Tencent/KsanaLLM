/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>

#include "ksana_llm/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BatchManager {
 public:
  explicit BatchManager(std::shared_ptr<Context> context, size_t max_pp_batch_num = 1);

  // Register a model instance to current batch manager.
  Status RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance);

  // Set the batch scheduler to batch manager.
  void SetBatchScheduler(std::shared_ptr<BatchSchedulerInterface> batch_scheduler);

  // Set the runtime instance to batch manager.
  void SetLlmRuntime(std::shared_ptr<LlmRuntime> llm_runtime);

  // Enqueue a request to waiting queue.
  Status Enqueue(std::shared_ptr<Request> &request);

  // Wait all requests done.
  Status WaitAllDone();

  // Process and get next running jobs.
  // Used for standalone mode, or master node of distributed mode.
  //
  // In distributed mode,
  // the master node is responsible for lookup embedding, layers forward, and the final lm head and sampling,
  // the worker node is responsible for layers forward only.
  Status MainProcess(size_t pp_batch_idx);

  // Process received request in distributed mode.
  // Used only for worker node of distributed mode.
  Status WorkerProcess();

  // Start the batch manager.
  Status Start();

  // Stop the batch manager.
  Status Stop();

 private:
  // The global context.
  std::shared_ptr<Context> context_;

  // The batch scheduler.
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_ = nullptr;

  // Maximum number of pipeline parallel batch threads
  size_t max_pp_batch_num_;

  // The master and worker threads.
  std::vector<std::unique_ptr<std::thread>> main_threads_;
  std::unique_ptr<std::thread> worker_thread_;

  // Whether batch manager should be stopped.
  std::atomic<bool> terminated_ = false;

  // The model name to model instance.
  std::unordered_map<std::string, std::shared_ptr<ModelInstance>> model_instances_;

  // The runtime instance.
  std::shared_ptr<LlmRuntime> llm_runtime_ = nullptr;

  // To guard result maintainer.
  std::mutex infer_reqs_maintainer_mutex_;

 private:
  ScheduleOutput MergeScheduleOutputGroup(std::shared_ptr<ScheduleOutputGroup> &schedule_output_group);
};

}  // namespace ksana_llm
