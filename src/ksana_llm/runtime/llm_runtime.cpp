/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"
#include <algorithm>
#include <execution>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
LlmRuntime::LlmRuntime(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                       std::shared_ptr<Context> context)
    : context_(context) {
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(),
                                                batch_scheduler_config.max_pp_batch_num, context_);

  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_.push_back(std::make_shared<Sampler>(batch_scheduler_config, worker_id, context_));
  }
  threadpool_ = std::make_shared<ThreadPool>(2);
  threadpool_->Start();
}

void LlmRuntime::SetCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers) {
  cache_managers_ = cache_managers;
}

void LlmRuntime::SetMultiBatchController(std::shared_ptr<MultiBatchController> controller) {
  multi_batch_controller_ = controller;
}

void LlmRuntime::SetDraftGenerator(std::shared_ptr<DraftGeneratorInterface> draft_generator) {
  if (draft_generator_ != nullptr) {
    KLLM_LOG_WARNING << "draft_generator already exists, currently only supports one, will replace the previous one";
  }
  draft_generator_ = draft_generator;
}

#ifdef ENABLE_CUDA
// In a CUDA environment, it's necessary to compute perfill and decode together.
// Therefore, kGroupStageMap is required to map to a single stage for synchronized processing.
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STATE_DECODE}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#else
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STAGE_CONTEXT}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#endif

void LlmRuntime::BuildForwardRequests(
    size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  PROFILE_EVENT_SCOPE(BuildForwardRequests, fmt::format("BuildForwardRequests_{}", multi_batch_id));
  // maybe async
  int logits_offset = 0;
  for (auto& req_ptr : reqs) {
    req_ptr->step += 1;
    req_ptr->logits_offset = logits_offset;
    logits_offset += req_ptr->sampling_token_num;

    ModelInstance* key = req_ptr->model_instance.get();
    InferStage grouped_stage = kGroupStageMap[req_ptr->infer_stage];

    if (grouped_reqs.find(key) == grouped_reqs.end()) {
      grouped_reqs[key] = {};
    }

    if (grouped_reqs[key].find(grouped_stage) == grouped_reqs[key].end()) {
      grouped_reqs[key][grouped_stage] = {};
    }

    ForwardRequest forward_req;
    BuildForwardRequestFromInferRequest(forward_req, req_ptr, req_ptr->model_instance->GetLayerNum(),
                                        key->GetLogitsPtr(multi_batch_id));
    grouped_reqs[key][grouped_stage].push_back(forward_req);
  }
}

void LlmRuntime::BuildForwardRequestFromInferRequest(ForwardRequest& forward_req,
                                                     std::shared_ptr<InferRequest>& req_ptr, uint32_t layer_num,
                                                     std::vector<float*> logits_buf) {
  forward_req.req_id = req_ptr->req_id;
  forward_req.infer_stage = req_ptr->infer_stage;
  forward_req.step = req_ptr->step;
  forward_req.kv_cached_token_num = req_ptr->kv_cached_token_num;
  forward_req.logits_custom_length = req_ptr->logits_custom_length;
  forward_req.sampling_token_num = req_ptr->sampling_token_num;
  forward_req.last_step_token_num = req_ptr->last_step_token_num;
  forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
  forward_req.logits_buf = logits_buf;
  forward_req.logits_offset = req_ptr->logits_offset;
  forward_req.request_target = &req_ptr->request_target;
  forward_req.response = &req_ptr->response;
  forward_req.draft_token_num = req_ptr->draft_tokens.size();
  forward_req.forwarding_tokens = &(req_ptr->forwarding_tokens);
  forward_req.flexible_cached_copy_tasks = &(req_ptr->flexible_cached_copy_tasks);
  forward_req.input_refit_embedding = &(req_ptr->input_refit_embedding);
  forward_req.mrotary_embedding_pos_offset = &(req_ptr->mrotary_embedding_pos_offset);
  forward_req.is_use_prefix_cache = req_ptr->is_use_prefix_cache;
  forward_req.flexible_cache_len = req_ptr->flexible_cached_copy_tasks.size();
  forward_req.prefix_cache_len = req_ptr->prefix_cache_len + forward_req.flexible_cache_len;
  forward_req.is_cudagraph_capture_request = req_ptr->is_cudagraph_capture_request;
  forward_req.sampling_config = &(req_ptr->sampling_config);
  forward_req.span_context = req_ptr->span_context;
  forward_req.timestamp_in_ms = req_ptr->timestamp_in_ms;
  forward_req.req_ctx = req_ptr->req_ctx;
  forward_req.cache_manager = req_ptr->cache_manager;
  forward_req.attn_dp_group_id = req_ptr->attn_dp_group_id;

#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
  forward_req.accepted_hidden_states_ptr = &(req_ptr->accepted_hidden_states);
  BuildFlatKVCacheBlkIds(layer_num, req_ptr->kv_cache_blocks, forward_req.atb_kv_cache_base_blk_ids,
                         req_ptr->cache_manager);
#endif
}

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<WorkerInferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  int logits_offset = 0;
  PROFILE_EVENT_SCOPE(BuildForwardRequests, "BuildForwardRequests");
  for (size_t i = 0; i < reqs.size(); ++i) {
    std::shared_ptr<WorkerInferRequest>& req_ptr = reqs[i];
    req_ptr->step += 1;
    ModelInstance* key = req_ptr->model_instance.get();
    InferStage grouped_stage = kGroupStageMap[req_ptr->infer_stage];

    if (grouped_reqs.find(key) == grouped_reqs.end()) {
      grouped_reqs[key] = {};
    }

    if (grouped_reqs[key].find(grouped_stage) == grouped_reqs[key].end()) {
      grouped_reqs[key][grouped_stage] = {};
    }

    ForwardRequest forward_req;
    forward_req.req_id = req_ptr->req_id;
    forward_req.infer_stage = req_ptr->infer_stage;
    forward_req.step = req_ptr->step;
    forward_req.kv_cached_token_num = req_ptr->kv_cached_token_num;
    forward_req.logits_custom_length = 0;
    forward_req.kv_cache_ptrs = req_ptr->GetBlockPtrs();
    forward_req.logits_buf = {};
    forward_req.logits_offset = ++logits_offset;
    forward_req.request_target = &req_ptr->request_target;
    forward_req.response = &req_ptr->response;
    forward_req.forwarding_tokens = &(req_ptr->forwarding_tokens);
    forward_req.flexible_cached_copy_tasks = &(req_ptr->flexible_cached_copy_tasks);
    forward_req.input_refit_embedding = &(req_ptr->input_refit_embedding);
    forward_req.mrotary_embedding_pos_offset = &(req_ptr->mrotary_embedding_pos_offset);
    forward_req.is_use_prefix_cache = req_ptr->is_use_prefix_cache;
    forward_req.flexible_cache_len = 0;
    forward_req.prefix_cache_len = req_ptr->prefix_cache_len + forward_req.flexible_cache_len;
    forward_req.is_cudagraph_capture_request = false;
    forward_req.timestamp_in_ms = 0;
    forward_req.cache_manager = req_ptr->cache_manager;
    forward_req.attn_dp_group_id = req_ptr->attn_dp_group_id;
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
    // forward_req.accepted_hidden_states_ptr = &(req_ptr->accepted_hidden_states);
    uint32_t layer_num = req_ptr->model_instance->GetLayerNum();
    BuildFlatKVCacheBlkIds(layer_num, req_ptr->kv_cache_blocks, forward_req.atb_kv_cache_base_blk_ids,
                           req_ptr->cache_manager);

#endif
    grouped_reqs[key][grouped_stage].emplace_back(std::move(forward_req));
  }
}

#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
// NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
// The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not
// guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num,
// head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which are
// independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
// block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To
// interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's
// self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management
// capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.
// 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
// pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
// should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
// head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
// layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
// 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
// pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
// as follows:
//    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
//    modification in step 1, cache_base_ptr.
//    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
//    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
//    layer_num * 2, b4 * layer_num * 2].
//    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
//    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
//    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
//    block_token_num, head_num, head_dim].
//    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
//    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
//    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
//    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
// More detail refer to docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md
void LlmRuntime::BuildFlatKVCacheBlkIds(uint32_t layer_num, const std::vector<std::vector<int>>& device_block_ids,
                                        std::vector<std::vector<int32_t>>& atb_block_ids,
                                        std::shared_ptr<CacheManagerInterface> cache_manager) {
  size_t rank_num = device_block_ids.size();
  atb_block_ids.resize(rank_num);
  for (size_t rank = 0; rank < rank_num; ++rank) {
    atb_block_ids[rank].clear();
    // for dedicate device kv blocks
    size_t base_id = cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank)->GetBlocksBaseId();
    atb_block_ids[rank].reserve(device_block_ids[rank].size());
    std::transform(device_block_ids[rank].begin(), device_block_ids[rank].end(),
                   std::back_inserter(atb_block_ids[rank]), [base_id, layer_num](int block_id) {
                     size_t original_block_id = block_id - base_id;
                     // NOTE(karlluo): only support bfloat16 or float16, so we just dedicate sizeof(float16) here
                     return original_block_id * layer_num * 2;
                   });
  }
}
#endif

Status LlmRuntime::RunSerially(
    size_t multi_batch_id,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(RunSerially_, fmt::format("RunSerially_{}_{}", multi_batch_id, epilogue));
  Status result_status = Status();
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      std::vector<std::future<Status>> inst_results =
          model_inst->ForwardAsync(multi_batch_id, worker_group_, stage, vec_req, epilogue, run_mode);
      for (auto& worker_result : inst_results) {
        Status status = worker_result.get();
        if (!status.OK()) {
          result_status = status;
        }
      }
    }
  }
  return result_status;
}

template <typename T>
void LlmRuntime::ReorderInferRequests(std::vector<std::shared_ptr<T>>& reqs) {
  PROFILE_EVENT_SCOPE(ReorderInferRequests, "ReorderInferRequests");
  // Due to the different calculation logic used for multi-token and single-token in the Attention layer,
  // the requests are first sorted to utilize contiguous space for accelerated inference.
  // Sort the infer_reqs list based on the number of tokens that need to be calculated for the KV cache.
  std::sort(reqs.begin(), reqs.end(), [this](const auto& a, const auto& b) {
    // For dp case, the order is: [group1_prefill, group2_prefill, group1_decode, group2_decode]
    const int a_token_num = a->forwarding_tokens.size() - a->kv_cached_token_num;
    const int b_token_num = b->forwarding_tokens.size() - b->kv_cached_token_num;

    const static size_t decode_threshold_len = IsAbsorbWeightsEnabled() ? 2 : 1;

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
}

Status LlmRuntime::Forward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs, bool epilogue,
                           RunMode run_mode) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(multi_batch_id, reqs, grouped_reqs);
  return AuxForward(multi_batch_id, grouped_reqs, epilogue, run_mode);
}

Status LlmRuntime::Forward(size_t multi_batch_id, std::vector<std::shared_ptr<WorkerInferRequest>>& reqs,
                           bool epilogue) {
  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_reqs;
  BuildForwardRequests(reqs, grouped_reqs);
  return AuxForward(multi_batch_id, grouped_reqs, epilogue);
}

Status LlmRuntime::AuxForward(
    size_t multi_batch_id,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(Forward_, fmt::format("Forward_{}_{}", multi_batch_id, epilogue));
  // context decode and decode run serially in single thread
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    // Wait all instances done and check status.
    auto ret = RunSerially(multi_batch_id, grouped_reqs, epilogue, run_mode);
    return ret;
  }

  std::vector<std::vector<std::future<Status>>> results;
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      results.push_back(model_inst->ForwardAsync(multi_batch_id, worker_group_, stage, vec_req, epilogue, run_mode));
    }
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& inst_results : results) {
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        result_status = status;
      }
    }
  }
  return result_status;
}

void LlmRuntime::BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::vector<SamplingRequest>& sampling_reqs) {
  PROFILE_EVENT_SCOPE(BuildSamplingRequest_, fmt::format("BuildSamplingRequest_{}", multi_batch_id));
  for (std::shared_ptr<InferRequest> req_ptr : reqs) {
    SamplingRequest sampling_req;
    sampling_req.req_id = req_ptr->req_id;
    sampling_req.logits_custom_length = req_ptr->logits_custom_length;
    sampling_req.input_tokens = &(req_ptr->input_tokens);
    sampling_req.forwarding_tokens = &(req_ptr->forwarding_tokens);
    sampling_req.sampling_token_num = req_ptr->sampling_token_num;
    sampling_req.last_step_token_num = req_ptr->last_step_token_num;
    sampling_req.sampling_result_tokens = &(req_ptr->sampling_result_tokens);
    sampling_req.sampling_result_tokens->clear();
    sampling_req.response = &(req_ptr->response);
    sampling_req.request_target = &(req_ptr->request_target);
    sampling_req.logprobs = &(req_ptr->logprobs);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->model_instance->GetLogitsPtr(multi_batch_id);
    sampling_req.sampling_config = &(req_ptr->sampling_config);
    sampling_req.req_group = &(req_ptr->req_group);
    sampling_req.req_ctx = req_ptr->req_ctx;
    if (sampling_req.sampling_config->num_beams > 1) {
      sampling_req.sampling_config->logprobs_num =
          std::max(sampling_req.sampling_config->logprobs_num, sampling_req.sampling_config->num_beams);
      sampling_req.sampling_config->topk =
          std::max(sampling_req.sampling_config->topk, sampling_req.sampling_config->num_beams);
    }
    sampling_req.ngram_dict = &(req_ptr->ngram_dict);
    sampling_req.model_config = &(req_ptr->model_instance->GetModelConfig());
    sampling_req.grammar_matcher = req_ptr->grammar_matcher;
    sampling_reqs.push_back(sampling_req);
  }
}

Status LlmRuntime::Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs) {
  PROFILE_EVENT_SCOPE(Sampling, fmt::format("Sampling_{}", multi_batch_id));
  std::vector<SamplingRequest> sampling_reqs;
  BuildSamplingRequest(multi_batch_id, reqs, sampling_reqs);

  std::vector<std::future<Status>> results;
  for (int worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group_->GetWorker(worker_id)->SamplingAsync(multi_batch_id, samplers_[worker_id], sampling_reqs));
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& result : results) {
    Status status = result.get();
    if (!status.OK()) {
      result_status = status;
    }
  }

  threadpool_->Submit([reqs]() mutable {
    const auto current_time = ProfileTimer::GetCurrentTimeInMs();
    std::for_each(std::execution::par_unseq, reqs.begin(), reqs.end(), [current_time](const auto& req) {
      if (req->infer_stage == STAGE_CONTEXT) {
        REPORT_METRIC(time_to_first_token_ms, current_time - req->timestamp_in_ms);
      }
    });
  });

  return result_status;
}

void LlmRuntime::DraftTokenFilter(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::vector<std::tuple<std::string, int, int>> reqs_tokens;
  for (auto& req : reqs) {
    if (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->draft_tokens.size()) {
      KLLM_LOG_ERROR << fmt::format(
          "req {} sampling_result_tokens.size = {}, mtp_draft_tokens.size = {}, trie_draft_tokens.size = {}",
          req->req_id, req->sampling_result_tokens.size(), req->draft_tokens.mtp.size(), req->draft_tokens.trie.size());
      continue;
    }

    // Check which tokens are predicted correctly.
    size_t draft_hit_num = 0;
    req->accepted_tokens.clear();
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
      if (req->sampling_result_tokens[i] != draft_tokens[i]) {
        break;
      }
      // stop if stop token
      if (std::find(req->sampling_config.stop_token_ids.begin(), req->sampling_config.stop_token_ids.end(),
                    draft_tokens[i]) != req->sampling_config.stop_token_ids.end()) {
        break;
      }
      ++draft_hit_num;
    }

    KLLM_LOG_DEBUG << "draft accepted: " << draft_hit_num << " / " << req->draft_tokens.size()
                   << ". samp: " << req->sampling_result_tokens << ", draft: " << draft_tokens;
    req->accepted_tokens.swap(draft_tokens);
    req->accepted_tokens.resize(draft_hit_num);
    req->generated_token = req->sampling_result_tokens[draft_hit_num];  // only kStepGenerateTokenNum(1) token now
    reqs_tokens.emplace_back(req->kv_comm_group_key, req->kv_comm_request_id, req->generated_token);
    req->sampling_result_tokens.clear();
    req->draft_tokens.clear();
  }
  TransferEngine::GetInstance()->Send(reqs_tokens);
}

Status LlmRuntime::MTPForward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                              const bool epilogue) {
  if (!mtp_forward_ || !context_->IsChief()) {
    return Status();
  }

  // select mtp req
  constexpr size_t kCompressKvMaxMtpLen = 1024;

  std::vector<std::shared_ptr<InferRequest>> mtp_reqs;
  mtp_reqs.reserve(reqs.size());
  for (const auto& req : reqs) {
    if (req->mtp_kv_cached_token_num != req->kv_cached_token_num) {
      continue;
    }
    if (req->grammar_matcher) {
      // TODO(ethanyczeng): Add MTP support
      continue;
    }
    mtp_reqs.emplace_back(req);
  }

  if (mtp_reqs.empty()) {
    KLLM_LOG_DEBUG << "skip mtp forward";
    return Status();
  }

  std::vector<int> first_tokens(mtp_reqs.size());  // store req first token
  for (size_t i = 0; i < mtp_reqs.size(); ++i) {
    auto& req = mtp_reqs[i];
    auto mtp_forward_token = req->GetVerifiedTokens();
    first_tokens[i] = mtp_forward_token.front();
    mtp_forward_token.erase(mtp_forward_token.begin());

    req->forwarding_tokens = std::move(mtp_forward_token);
    req->forwarding_tokens_draft_num = req->accepted_tokens.size();  // already removed wrong token
    req->sampling_token_num = kStepGenerateTokenNum;
    req->last_step_token_num = kStepGenerateTokenNum;
    req->kv_cached_token_num = req->mtp_kv_cached_token_num;
    req->prefix_cache_len = req->kv_cached_token_num;
  }

  ReorderInferRequests(mtp_reqs);
  Forward(multi_batch_id, mtp_reqs, epilogue, RunMode::kNextN);
  Sampling(multi_batch_id, mtp_reqs);

  for (size_t i = 0; i < mtp_reqs.size(); ++i) {
    auto& req = mtp_reqs[i];
    req->forwarding_tokens.resize(req->forwarding_tokens.size() - kStepGenerateTokenNum);
    req->forwarding_tokens.insert(req->forwarding_tokens.begin(), first_tokens[i]);
    req->draft_tokens.mtp = std::move(req->sampling_result_tokens);

    req->mtp_kv_cached_token_num = req->forwarding_tokens.size();
  }
  return Status();
}

void LlmRuntime::GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  if (draft_generator_ == nullptr) {
    return;
  }
  PROFILE_EVENT_SCOPE(GenerateDraftToken_, fmt::format("GenerateDraftToken"));
  for (auto& req : reqs) {
    if (req->grammar_matcher) {
      // TODO(ethanyczeng): Add MTP support
      continue;
    }
    std::vector<int> tokens;
    tokens.reserve(req->forwarding_tokens.size() - req->forwarding_tokens_draft_num + req->accepted_tokens.size() +
                   kStepGenerateTokenNum + req->draft_tokens.mtp.size());
    tokens.insert(tokens.end(), req->forwarding_tokens.begin(),
                  req->forwarding_tokens.end() - req->forwarding_tokens_draft_num);
    tokens.insert(tokens.end(), req->accepted_tokens.begin(), req->accepted_tokens.end());
    tokens.emplace_back(req->generated_token);
    tokens.insert(tokens.end(), req->draft_tokens.mtp.begin(), req->draft_tokens.mtp.end());
    draft_generator_->GenerateDraft(tokens, req->step, req->suggested_draft_num, req->draft_tokens.trie,
                                    req->draft_tokens.mtp.size(), req->accepted_tokens.size(), req->req_id);
  }
}

Status LlmRuntime::Step(ScheduleOutput* schedule_output, bool epilogue) {
  if (context_->IsChief()) {
    return StepOnChief(schedule_output, epilogue);
  }
  return StepOnWorker(schedule_output, epilogue);
}

Status LlmRuntime::StepOnChief(ScheduleOutput* schedule_output, bool epilogue) {
  KLLM_LOG_DEBUG << "Enter llm runtime StepOnChief. multi_batch_id=" << schedule_output->multi_batch_id
                 << ", epilogue=" << epilogue;
  PROFILE_EVENT_SCOPE(StepOnChief_, fmt::format("StepOnChief_{}_{}", schedule_output->multi_batch_id, epilogue));

  std::shared_ptr<ModelInstance> model_instance = schedule_output->running_reqs[0]->model_instance;
  ReorderInferRequests(schedule_output->running_reqs);

  if (!epilogue) {
    // Alloc resources before forwarding
    model_instance->AllocResources(schedule_output->multi_batch_id);
  }
  // Inference forward.
  time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
  if (epilogue && !context_->IsStandalone()) {
    auto next_multi_batch_id = multi_batch_controller_->NotifyOtherBatchCanRun();
    KLLM_LOG_DEBUG << "unlock multi_batch_id=" << schedule_output->multi_batch_id << ", start to recv hiddens";
    multi_batch_controller_->WaitUtilCanRecvCurrentHiddenUnits(schedule_output->multi_batch_id, next_multi_batch_id);
    SetHiddenUnitMeta(schedule_output->multi_batch_id, schedule_output->running_reqs, model_instance);
    RecvHiddenUnits(schedule_output->multi_batch_id);

    KLLM_LOG_DEBUG << "try to lock again multi_batch_id=" << schedule_output->multi_batch_id << ", epilogue=true";
    multi_batch_controller_->WaitUtilCurrentBatchCanRun(schedule_output->multi_batch_id);
  }
  Forward(schedule_output->multi_batch_id, schedule_output->running_reqs, epilogue);
  time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
  KLLM_LOG_DEBUG << "LlmRuntime Forward multi_batch_id=" << schedule_output->multi_batch_id << ", epilogue=" << epilogue
                 << ", time cost=" << end_time_ms - start_time_ms << "ms";

  // Sampling only in standalone mode or epilogue=true in distributed mode
  if (context_->IsStandalone() || epilogue) {
    PROFILE_EVENT_SCOPE(SamplingAndMTP_, fmt::format("SamplingAndMTP_{}", schedule_output->multi_batch_id));
    Sampling(schedule_output->multi_batch_id, schedule_output->running_reqs);
    DraftTokenFilter(schedule_output->running_reqs);
    MTPForward(schedule_output->multi_batch_id, schedule_output->running_reqs, epilogue);
    GenerateDraftToken(schedule_output->running_reqs);

    // Forwarding finished, free resources.
    model_instance->FreeResources(schedule_output->multi_batch_id);
    // Note(TJ): donot need NotifyOtherBatchCanRun, because maybe this batch will enter again.
    KLLM_LOG_DEBUG << "finish multi_batch_id=" << schedule_output->multi_batch_id << ", epilogue=" << epilogue;
  }
  KLLM_LOG_DEBUG << "Leave llm runtime StepOnChief. multi_batch_id=" << schedule_output->multi_batch_id
                 << ", epilogue=" << epilogue;
  return Status();
}

Status LlmRuntime::StepOnWorker(ScheduleOutput* schedule_output, bool epilogue) {
  KLLM_LOG_DEBUG << "llm runtime StepOnWorker invoked multi_batch_id=" << schedule_output->multi_batch_id;
  PROFILE_EVENT_SCOPE(StepOnWorker_, fmt::format("StepOnWorker_{}_{}", schedule_output->multi_batch_id, epilogue));
  // Worker always pass result to next step
  KLLM_CHECK(epilogue == false);

  ReorderInferRequests(schedule_output->worker_running_reqs);

  for (size_t dp_swapout_req_block_ids_idx = 0;
       dp_swapout_req_block_ids_idx < schedule_output->swapout_req_block_ids.size(); ++dp_swapout_req_block_ids_idx) {
    std::stringstream so_ss;
    if (!schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].empty()) {
      for (auto it = schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].begin();
           it != schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].end(); ++it) {
        so_ss << it->first << ", ";
        Status status =
            cache_managers_[dp_swapout_req_block_ids_idx]->SwapoutRequestMemoryBlockAsync(it->first, it->second);
        if (!status.OK()) {
          return status;
        }
      }
      KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                     << ", dp_idx=" << dp_swapout_req_block_ids_idx << ", SwapoutRequestMemoryBlockAsync req_ids=("
                     << so_ss.str() << ")";
    }
  }

  for (size_t dp_merged_swapout_req_ids_idx = 0;
       dp_merged_swapout_req_ids_idx < schedule_output->merged_swapout_req_ids.size();
       ++dp_merged_swapout_req_ids_idx) {
    auto& dp_merged_swapout_req_ids = schedule_output->merged_swapout_req_ids[dp_merged_swapout_req_ids_idx];
    if (dp_merged_swapout_req_ids.empty()) {
      continue;
    }
    KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                   << ", WaitSwapoutRequestMemoryBlock dp_idx=" << dp_merged_swapout_req_ids_idx
                   << ", req num=" << dp_merged_swapout_req_ids.size()
                   << ", ids=" << Vector2Str(dp_merged_swapout_req_ids);
    cache_managers_[dp_merged_swapout_req_ids_idx]->WaitSwapoutRequestMemoryBlock(dp_merged_swapout_req_ids);
  }

  for (size_t dp_swapin_req_block_ids_idx = 0;
       dp_swapin_req_block_ids_idx < schedule_output->swapin_req_block_ids.size(); ++dp_swapin_req_block_ids_idx) {
    std::stringstream si_ss;
    if (!schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].empty()) {
      for (auto it = schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].begin();
           it != schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].end(); ++it) {
        si_ss << it->first << ", ";

        Status status =
            cache_managers_[dp_swapin_req_block_ids_idx]->SwapinRequestMemoryBlockAsync(it->first, it->second);
        if (!status.OK()) {
          return status;
        }
      }
      KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                     << ", dp_idx=" << dp_swapin_req_block_ids_idx << ", SwapinRequestMemoryBlockAsync req_ids=("
                     << si_ss.str() << ")";
    }
  }

  for (size_t dp_merged_swapin_req_ids_idx = 0;
       dp_merged_swapin_req_ids_idx < schedule_output->merged_swapin_req_ids.size(); ++dp_merged_swapin_req_ids_idx) {
    auto& dp_merged_swapin_req_ids = schedule_output->merged_swapin_req_ids[dp_merged_swapin_req_ids_idx];
    if (dp_merged_swapin_req_ids.empty()) {
      continue;
    }
    KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                   << ", WaitSwapinRequestMemoryBlock dp_idx=" << dp_merged_swapin_req_ids_idx
                   << ", req num=" << dp_merged_swapin_req_ids.size()
                   << ", ids=" << Vector2Str(dp_merged_swapin_req_ids);
    cache_managers_[dp_merged_swapin_req_ids_idx]->WaitSwapinRequestMemoryBlock(dp_merged_swapin_req_ids);
  }

  std::shared_ptr<ModelInstance> model_instance = schedule_output->worker_running_reqs[0]->model_instance;

  // Inference forward.
  model_instance->AllocResources(schedule_output->multi_batch_id);

  SetHiddenUnitMeta(schedule_output->multi_batch_id, schedule_output->worker_running_reqs, model_instance);
  RecvHiddenUnits(schedule_output->multi_batch_id);
  Forward(schedule_output->multi_batch_id, schedule_output->worker_running_reqs, epilogue);
  model_instance->FreeResources(schedule_output->multi_batch_id);
  return Status();
}

template void LlmRuntime::ReorderInferRequests<InferRequest>(std::vector<std::shared_ptr<InferRequest>> &reqs);
template void LlmRuntime::ReorderInferRequests<WorkerInferRequest>(
    std::vector<std::shared_ptr<WorkerInferRequest>> &reqs);

}  // namespace ksana_llm
