/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>

#include "ksana_llm/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"
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
  explicit BatchManager(const RuntimeConfig &runtime_config, std::shared_ptr<Context> context);

  // Register a model instance to current batch manager.
  Status RegisterModelInstance(const std::shared_ptr<ModelInstance> &model_instance);

  // Set the batch scheduler to batch manager.
  void SetBatchScheduler(std::shared_ptr<BatchSchedulerInterface> batch_scheduler);

  // Set the runtime instance to batch manager.
  void SetLlmRuntime(std::shared_ptr<LlmRuntime> llm_runtime);

  // Set the multi_batch contorller instance to batch manager.
  void SetMultiBatchController(std::shared_ptr<MultiBatchController> controller);

  // Enqueue a request to waiting queue.
  Status Enqueue(std::shared_ptr<Request> &request);

  Status ProcessScheduleData(
      const std::pair<std::shared_ptr<ScheduleOutput>,
                      std::pair<std::shared_ptr<std::unordered_map<
                                    ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>>>,
                                std::shared_ptr<std::vector<SamplingRequest>>>> &schedule_data);

  // Wait all requests done.
  Status WaitAllDone();

  // Process and get next running jobs.
  // Used for standalone mode, or master node of distributed mode.
  //
  // In distributed mode,
  // the master node is responsible for lookup embedding, layers forward, and the final lm head and sampling,
  // the worker node is responsible for layers forward only.
  Status MainProcess(size_t multi_batch_id);

  // Process received request in distributed mode.
  // Used only for worker node of distributed mode.
  Status WorkerProcess();

  // Start the batch manager.
  Status Start();

  // Stop the batch manager.
  Status Stop();

 private:
  std::vector<size_t> GetHiddenUnitShape(ScheduleOutput *schedule_output);

 private:
  RuntimeConfig runtime_config_;

  // The global context.
  std::shared_ptr<Context> context_;

  // The batch scheduler.
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_ = nullptr;

  // The multi batch controllor.
  std::shared_ptr<MultiBatchController> multi_batch_controller_ = nullptr;

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
