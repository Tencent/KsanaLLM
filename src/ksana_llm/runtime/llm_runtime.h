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
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/absorb_weights_type.h"
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

  void SetAsync(const bool is_enable) { enable_async_ = is_enable; }

  // Execute one schedule output in parallel.
  // epilogue is used only for distributed master node, to process lm head and sampler.
  Status Step(ScheduleOutput *schedule_output,
              std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
              std::vector<SamplingRequest> &sampling_reqs, bool epilogue);

  // Reorder the infer_request list, placing the requests from the Multi-Token Forwarding at the front
  // and the requests from the Single-Token Forwarding at the back.
  template <typename T>
  void ReorderInferRequests(std::vector<std::shared_ptr<T>> &reqs) {
    PROFILE_EVENT_SCOPE(ReorderInferRequests, "ReorderInferRequests");
    // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
    // the requests are first sorted to utilize contiguous space for accelerated inference.
    // Sort the infer_reqs list based on the number of tokens that need to be calculated for the KV cache.
    std::sort(reqs.begin(), reqs.end(), [this](const auto &a, const auto &b) {
      // For dp case, the order is: [group1_prefill, group2_prefill, group1_decode, group2_decode]
      const int a_token_num = a->forwarding_tokens.size() - a->kv_cached_token_num;
      const int b_token_num = b->forwarding_tokens.size() - b->kv_cached_token_num;

      const static size_t decode_threshold_len = IsAbsorbWeightsEnabled() && IsAbsorbWeightsEnabled() ? 2 : 1;

      const bool is_a_decode = a_token_num <= decode_threshold_len && a->kv_cached_token_num != 0;
      const bool is_b_decode = b_token_num <= decode_threshold_len && b->kv_cached_token_num != 0;

      // Both prefill or decode, the a_token_num or b_token_num may be zero.
      if (is_a_decode == is_b_decode) {
        if (a->attn_dp_group_id != b->attn_dp_group_id) {
          return a->attn_dp_group_id < b->attn_dp_group_id;
        } else {
          if (a_token_num != b_token_num) {
            return a_token_num > b_token_num;
          }
          if (a->kv_cached_token_num != b->kv_cached_token_num) {
            return a->kv_cached_token_num < b->kv_cached_token_num;
          }
          return a->req_id < b->req_id;
        }
      } else {
        // One is prefill, another is decode, prefill before decode
        return !is_a_decode;
      }
    });

    // reset logits offset after reorder
    if constexpr (std::is_same_v<T, InferRequest>) {
      size_t logits_offset = 0;
      for (auto &req : reqs) {
        req->logits_offset = logits_offset;
        logits_offset += req->sampling_token_num;
      }
    }
  }

  // Build forward request, group by model name and stage, for distributed worker node.
  void BuildForwardRequests(
      std::vector<std::shared_ptr<WorkerInferRequest>> &reqs,
      std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs);

  // Build forward request, group by model name and stage.
  void BuildForwardRequests(
      size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
      std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs);

  // Build sampling request.
  void BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                            std::vector<SamplingRequest> &sampling_reqs, bool enable_main_layers_sampler = true);

  void DeepCopyAndSyncSamplingRequests(const std::vector<std::shared_ptr<InferRequest>> &running_reqs,
                                       std::vector<SamplingRequest> &sampling_reqs);

  // Execute the forward.
  Status Forward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs, bool epilogue,
                 std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
                 RunMode run_mode = RunMode::kMain);

  // Execute the forward, for distributed worker node.
  Status Forward(size_t multi_batch_id, std::vector<std::shared_ptr<WorkerInferRequest>> &reqs, bool epilogue,
                 std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs);

 private:
  // Execute the sampling.
  Status Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                  std::vector<SamplingRequest> &sampling_reqs, bool enable_main_layers_sampler = true);

  // Run multi-token and single-token serially in single thread.
  Status RunSerially(size_t multi_batch_id,
                     std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
                     bool epilogue, RunMode run_mode = RunMode::kMain);

  // A assisant of forward.
  Status AuxForward(size_t multi_batch_id,
                    std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
                    bool epilogue, RunMode run_mode = RunMode::kMain);

  void DraftTokenFilter(std::vector<std::shared_ptr<InferRequest>> &reqs);

  Status MTPForward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs, const bool epilogue,
                    std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> grouped_reqs,
                    std::vector<SamplingRequest> &sampling_reqs);

  void GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>> &reqs);

  void TransferGeneratedToken(std::vector<std::shared_ptr<InferRequest>> &reqs,
                              std::shared_ptr<TransferEngine> transfer_engine = TransferEngine::GetInstance());

  Status StepOnChief(ScheduleOutput *schedule_output,
                     std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
                     std::vector<SamplingRequest> &sampling_reqs, bool epilogue);
  Status StepOnWorker(ScheduleOutput *schedule_output,
                      std::map<ModelInstance *, std::map<InferStage, std::vector<ForwardRequest *>>> &grouped_reqs,
                      std::vector<SamplingRequest> &sampling_reqs, bool epilogue);

 private:
  bool mtp_forward_ = false;

  bool enable_async_ = false;

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
