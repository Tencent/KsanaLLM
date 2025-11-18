/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/multi_batch_controller/multi_batch_controller.h"
#include "ksana_llm/runtime/draft_generator/draft_generator_controller.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
class GenerationController;

class LlmRuntime {
 public:
  LlmRuntime(const BatchSchedulerConfig &batch_scheduler_config, const RuntimeConfig &runtime_config,
             std::shared_ptr<Context> context);
  ~LlmRuntime() { threadpool_.Stop(); }

  // Set cache manager, used to operate the kv cache block.
  void SetCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers);

  // Set the multi_batch contorller.
  void SetMultiBatchController(std::shared_ptr<MultiBatchController> controller);

  // Set draft generator
  void SetDraftGeneratorController(std::shared_ptr<DraftGeneratorController> controller);

  // Set generation controller
  void SetGenerationController(std::shared_ptr<GenerationController> controller) {
    generation_controller_ = controller;
  }

  // Execute one schedule output in parallel.
  // epilogue is used only for distributed master node, to process lm head and sampler.
  Status Step(ScheduleOutput *schedule_output, std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs,
              std::vector<SamplingRequest> &sampling_reqs, bool epilogue);

  // Reorder the infer_request list, placing the requests from the Multi-Token Forwarding at the front
  // and the requests from the Single-Token Forwarding at the back.
  template <typename T>
  void ReorderInferRequests(std::vector<T> &reqs) {
    PROFILE_EVENT_SCOPE(ReorderInferRequests, "ReorderInferRequests");
    // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
    // the requests are first sorted to utilize contiguous space for accelerated inference.
    std::sort(reqs.begin(), reqs.end(), [this](const auto &a, const auto &b) {
      // For dp case, the order is: [group1_prefill, group1_decode, group2_prefill, group2_decode, ...]
      // NOTE: prefill may appear after decode (if they are in different dp groups)
      if (a->attn_dp_group_id != b->attn_dp_group_id) {
        return a->attn_dp_group_id < b->attn_dp_group_id;
      }

      auto get_vec_size = [](const auto &item) {
        if constexpr (std::is_same_v<std::decay_t<decltype(item)>, std::vector<int>>) {
          return item.size();
        } else {
          return item->size();
        }
      };

      // For non-dp case, the order is: [prefill, decode]
      const size_t a_token_num = get_vec_size(a->forwarding_tokens) - a->kv_cached_token_num;
      const size_t b_token_num = get_vec_size(b->forwarding_tokens) - b->kv_cached_token_num;

      const bool is_a_decode = a_token_num <= GetDecodeTokenNumThreshold() && a->kv_cached_token_num > 0;
      const bool is_b_decode = b_token_num <= GetDecodeTokenNumThreshold() && b->kv_cached_token_num > 0;

      if (is_a_decode == is_b_decode) {
        // Both prefill or decode, sort the infer_reqs list based on a_token_num and b_token_num,
        // i.e., the number of tokens that need to be calculated for the KV cache.
        // NOTE: a_token_num or b_token_num may be zero
        if (a_token_num != b_token_num) {
          return a_token_num > b_token_num;
        }
        if (a->kv_cached_token_num != b->kv_cached_token_num) {
          return a->kv_cached_token_num < b->kv_cached_token_num;
        }
        return a->req_id < b->req_id;
      } else {
        // One is prefill, another is decode, prefill before decode
        return !is_a_decode;
      }
    });

    // reset logits offset after reorder
    if constexpr (!std::is_same_v<std::decay_t<decltype(*std::declval<T>())>, WorkerInferRequest>) {
      size_t logits_offset = 0;
      for (auto &req : reqs) {
        req->logits_offset = logits_offset;
        logits_offset += req->sampling_token_num;
      }
    }
  }

  // Build forward request, group by model name and stage, for distributed worker node.
  void BuildForwardRequests(std::vector<std::shared_ptr<WorkerInferRequest>> &reqs,
                            std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs);

  // Build forward request, group by model name and stage.
  void BuildForwardRequests(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                            std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs);

  // Build sampling request.
  void BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                            std::vector<SamplingRequest> &sampling_reqs, bool enable_main_layers_sampler = true);

  void DeepCopyAndSyncSamplingRequests(const std::vector<std::shared_ptr<InferRequest>> &running_reqs,
                                       std::vector<SamplingRequest> &sampling_reqs);

  virtual void Forward(size_t multi_batch_id, std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs,
                       bool epilogue, RunMode run_mode = RunMode::kMain);

 private:
  // Execute the sampling.
  virtual void Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>> &reqs,
                        std::vector<SamplingRequest> &sampling_reqs, bool enable_main_layers_sampler = true);

  std::shared_ptr<WaitGroup> PrepareMtpInfoAsync(const std::vector<std::shared_ptr<InferRequest>> &reqs);
  Status MtpForward(const size_t multi_batch_id, std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs,
                    std::vector<SamplingRequest> &sampling_reqs, std::vector<std::shared_ptr<InferRequest>> &reqs,
                    const bool epilogue);

  void GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>> &reqs);

  void TransferGeneratedToken(std::vector<std::shared_ptr<InferRequest>> &reqs,
                              std::shared_ptr<TransferEngine> transfer_engine = TransferEngine::GetInstance());

  Status StepOnChief(ScheduleOutput *schedule_output,
                     std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs,
                     std::vector<SamplingRequest> &sampling_reqs, bool epilogue);
  Status StepOnWorker(ScheduleOutput *schedule_output,
                      std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs,
                      std::vector<SamplingRequest> &sampling_reqs, bool epilogue);

 private:
  const size_t mtp_step_num_ = 0;
  struct MtpPrepareData {
    std::shared_ptr<std::vector<int>> tokens;
    InferRequest *infer_req;
    size_t order_pos;
  };
  std::unordered_map<decltype(ForwardRequest::req_id), MtpPrepareData> mtp_prepared_data_;

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

  std::shared_ptr<DraftGeneratorController> draft_generator_controller_ = nullptr;

  std::shared_ptr<GenerationController> generation_controller_ = nullptr;

  ThreadPool threadpool_;
};

}  // namespace ksana_llm
