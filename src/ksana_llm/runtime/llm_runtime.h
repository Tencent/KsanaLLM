/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"
#include "ksana_llm/runtime/draft_generator/draft_generator_interface.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class LlmRuntime {
 public:
  LlmRuntime(const BatchSchedulerConfig &batch_scheduler_config, const RuntimeConfig &runtime_config,
             std::shared_ptr<Context> context);
  ~LlmRuntime() {
    if (threadpool_) {
      threadpool_->Stop();
    }
  }

  // Set cache manager, used to operate the kv cache block.
  void SetCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers);

  // Set the multi_batch contorller.
  void SetMultiBatchController(std::shared_ptr<MultiBatchController> controller);

  // Set draft generator
  void SetDraftGenerator(std::shared_ptr<DraftGeneratorInterface> draft_generator);

  void SetMtpForward(const bool is_enable) { mtp_forward_ = is_enable; }

  // Execute one schedule output in parallel.
  // epilogue is used only for distributed master node, to process lm head and sampler.
  Status Step(ScheduleOutput *schedule_output, bool epilogue);

  // Execute the forward.
  Status Forward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs, bool epilogue,
                 RunMode run_mode = RunMode::kMain);

  // Reorder the infer_request list, placing the requests from the Multi-Token Forwarding at the front
  // and the requests from the Single-Token Forwarding at the back.
  template <typename T>
  void ReorderInferRequests(std::vector<std::shared_ptr<T>> &reqs);

  // TODO(robertyuan): move static funtions to other place
  static void BuildForwardRequestFromInferRequest(ForwardRequest &forward_req, std::shared_ptr<InferRequest> &infer_req,
                                                  uint32_t layer_num, std::vector<float *> logits_buf);

#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
  // Build ATB KV cache block ids
  static void BuildFlatKVCacheBlkIds(uint32_t layer_num, const std::vector<std::vector<int>> &device_block_ids,
                                     std::vector<std::vector<int32_t>> &atb_block_ids,
                                     std::shared_ptr<CacheManagerInterface> cache_manager);
#endif

 private:
  // Execute the forward, for distributed worker node.
  Status Forward(size_t multi_batch_id, std::vector<std::shared_ptr<WorkerInferRequest>> &reqs, bool epilogue);

  // Execute the sampling.
  Status Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs);

  // Build forward request, group by model name and stage.
  void BuildForwardRequests(
      size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs);

  // Build forward request, group by model name and stage, for distributed worker node.
  void BuildForwardRequests(
      std::vector<std::shared_ptr<WorkerInferRequest>> &reqs,
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs);

  // Build sampling request.
  void BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                            std::vector<SamplingRequest> &sampling_reqs);


  // Run multi-token and single-token serially in single thread.
  Status RunSerially(
      size_t multi_batch_id,
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs,
      bool epilogue, RunMode run_mode = RunMode::kMain);

  // A assisant of forward.
  Status AuxForward(
      size_t multi_batch_id,
      std::unordered_map<ModelInstance *, std::unordered_map<InferStage, std::vector<ForwardRequest>>> &grouped_reqs,
      bool epilogue, RunMode run_mode = RunMode::kMain);

  void DraftTokenFilter(std::vector<std::shared_ptr<InferRequest>> &reqs);

  Status MTPForward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs, const bool epilogue);

  void GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>> &reqs);

  Status StepOnChief(ScheduleOutput *schedule_output, bool epilogue);
  Status StepOnWorker(ScheduleOutput *schedule_output, bool epilogue);

 private:
  bool mtp_forward_ = false;
  bool enable_flash_mla_ = false;

  // The cache manager inference used for inference engine.
  std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers_;

  // The multi batch controllor.
  std::shared_ptr<MultiBatchController> multi_batch_controller_ = nullptr;

  // The runtime context.
  std::shared_ptr<Context> context_ = nullptr;

  // The worker group for this runtime, do we need several worker_group?
  std::shared_ptr<WorkerGroup> worker_group_ = nullptr;

  // The sampler instance on every device.
  std::vector<std::shared_ptr<Sampler>> samplers_;

  std::shared_ptr<DraftGeneratorInterface> draft_generator_ = nullptr;

  // Threadpool used to metrics report.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;
};

}  // namespace ksana_llm
