/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>
#include <sstream>
#include <vector>
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {
InferRequest::InferRequest(std::shared_ptr<Request> &request, int index)
    : req_id(request->req_ids[index]),
      model_name(request->model_name),
      logits_custom_length(request->logits_custom_length),
      input_tokens(request->input_tokens),
      input_refit_embedding(request->input_refit_embedding),
      output_tokens(std::get<0>(request->output_group[index])),
      logprobs(std::get<1>(request->output_group[index])),
      request_target(request->request_target),
      response(request->response),
      cumulative_score(0),
      sampling_config(request->sampling_config),
      structured_generator_config(request->structured_generator_config),
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      abort_waiter(request->abort_waiter),
      finished(request->finisheds[index]),
      aborted(request->aborted),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      kv_comm_request_id(request->kv_comm_request_id),
      kv_comm_group_key(request->kv_comm_group_key),
      beam_search_group(request->beam_search_group),
      is_cudagraph_capture_request(request->is_cudagraph_capture_request),
      timestamp_in_ms(request->timestamp_in_ms),
      req_ctx(request->req_ctx) {}

InferRequest::~InferRequest() { KLLM_LOG_DEBUG << "req " << req_id << " destroyed."; }

std::string InferRequest::PrintKVBlockIds(bool print_details) const {
  std::ostringstream ss;
  ss << ", kv_cache_blocks_size:" << kv_cache_blocks.size() << ", kv_cache_blocks: {";
  for (size_t i = 0; i < kv_cache_blocks.size(); i++) {
    const auto &blocks = kv_cache_blocks[i];
    ss << "," << i << "=size(" << kv_cache_blocks[i].size() << ")";
    if (print_details) {
      ss << "{ ";
      for (auto blk_id : blocks) {
        ss << blk_id << ", ";
      }
      ss << "}, ";
    }
  }
  ss << "}";
  return ss.str();
}

std::string InferRequest::ToString(bool print_details) const {
  std::ostringstream oss;
  oss << " req(req_id:" << req_id << ", step:" << step << ", sampling_token_num:" << sampling_token_num
      << ", kv_cached_token_num:" << kv_cached_token_num << ", mtp_kv_cached_token_num:" << mtp_kv_cached_token_num
      << ", prefix_cache_len:" << prefix_cache_len << ", input_tokens_size:" << input_tokens.size()
      << ", output_tokens_size:" << output_tokens.size() << ", forwarding_tokens_size:" << forwarding_tokens.size()
      << ", draft_tokens_size:" << draft_tokens.size() << ", accepted_tokens_size:" << accepted_tokens.size()
      << ", generated_token:" << generated_token << PrintKVBlockIds(print_details) << ", swap_pending:" << swap_pending
      << ", finished:" << finished << ", aborted:" << aborted << ", finish_status:" << finish_status.ToString()
      << " ) ";
  return oss.str();
}

std::ostream &operator<<(std::ostream &os, const InferRequest &req) {
  os << req.ToString();
  return os;
}

void InferRequest::Notify() {
  for (size_t i = 0; i < req_group.size(); i++) {
    if (!req_group[i]->finished) return;
  }

  if (sampling_config.num_beams > 1) {
    std::sort(beam_search_group.begin(), beam_search_group.end(),
              [](const OutputTuple &a, const OutputTuple &b) { return std::get<2>(a) > std::get<2>(b); });

    for (size_t i = 0; i < req_group.size() && i < beam_search_group.size(); i++) {
      req_group[i]->output_tokens = std::move(std::get<0>(beam_search_group[i]));
      req_group[i]->logprobs = std::move(std::get<1>(beam_search_group[i]));
    }
  }

  for (size_t i = 0; i < req_group.size(); i++) {
    req_group[i]->ClearReqGroup();
  }

  // After a notification, the corresponding request may be destructed.
  // So we return early to avoid accessing any variables referencing it.
  if (aborted) {
    abort_waiter->Notify();
    return;
  }
  if (waiter) {
    waiter->Notify();
    return;
  }
  if (step_waiter) {
    step_waiter->Notify();
  }
}

void InferRequest::NotifyStep() {
  if (sampling_config.num_beams > 1) {
    int output_tokens_len = -1;
    for (size_t i = 0; i < req_group.size(); i++) {
      if (req_group[i]->finished) continue;
      output_tokens_len = output_tokens_len == -1 ? req_group[i]->output_tokens.size() : output_tokens_len;
      if (req_group[i]->output_tokens.size() != (size_t)output_tokens_len) return;
    }
  }

  if (step_waiter) {
    step_waiter->Notify();
  }
}

std::vector<int> InferRequest::GetVerifiedTokens() {
  std::vector<int> tokens = forwarding_tokens;
  tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num);
  tokens.insert(tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
  tokens.emplace_back(generated_token);
  return tokens;
}

std::vector<std::vector<void *>> InferRequest::GetBlockPtrs() {
  std::vector<std::vector<void *>> block_ptrs;
  block_ptrs.reserve(kv_cache_blocks.size());
  for (size_t rank = 0; rank < kv_cache_blocks.size(); ++rank) {
    std::vector<void *> block_ptr(kv_cache_blocks[rank].size());
    cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank)->GetBlockPtrs(kv_cache_blocks[rank],
                                                                                         block_ptr);
    block_ptrs.emplace_back(std::move(block_ptr));
  }
  return block_ptrs;
}

std::vector<int> InferRequest::GetKVOccupiedDevices() {
  std::vector<int> kv_occupied_devices;
  kv_occupied_devices = cache_manager->GetBlockAllocatorGroup()->GetBlockAllocatorDevices();
  KLLM_LOG_DEBUG << "req_id: " << kv_comm_request_id << ", kv_occupied_devices: " << Vector2Str(kv_occupied_devices)
                 << ".";
  return kv_occupied_devices;
}

ForwardRequest *InferRequest::GetForwardRequest(const std::vector<float *> &logits_buf) {
  if (forward_request_ == nullptr || reset_forward_request_) {
    reset_forward_request_ = false;
    forward_request_ = std::make_unique<ForwardRequest>();
    forward_request_->req_id = req_id;
    forward_request_->req_ctx = req_ctx;
    forward_request_->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward_request_->input_refit_embedding = &input_refit_embedding;
    forward_request_->mrotary_embedding_pos_offset = &mrotary_embedding_pos_offset;
    forward_request_->response = &response;
    forward_request_->origin_tokens = &forwarding_tokens;
    forward_request_->sampling_config = &sampling_config;
    forward_request_->forwarding_tokens =
        std::shared_ptr<decltype(forwarding_tokens)>(&forwarding_tokens, [](decltype(forwarding_tokens) *) {});
    forward_request_->request_target = std::make_shared<const std::map<std::string, TargetDescribe>>(request_target);
    forward_request_->cache_manager = cache_manager;

    const size_t rank_num = kv_cache_blocks.size();
    forward_request_->kv_cache_ptrs.resize(rank_num);
    forward_request_->atb_kv_cache_base_blk_ids.resize(rank_num);
  }

  forward_request_->infer_stage = infer_stage;
  forward_request_->step = step;
  forward_request_->kv_cached_token_num = kv_cached_token_num;
  forward_request_->logits_custom_length = logits_custom_length;
  forward_request_->sampling_token_num = sampling_token_num;
  forward_request_->last_step_token_num = last_step_token_num;
  forward_request_->logits_buf = logits_buf;
  forward_request_->logits_offset = logits_offset;
  forward_request_->draft_token_num = draft_tokens.size();
  forward_request_->is_use_prefix_cache = is_use_prefix_cache;
  forward_request_->prefix_cache_len = prefix_cache_len + flexible_cached_copy_tasks.size();
  forward_request_->is_cudagraph_capture_request = is_cudagraph_capture_request;
  forward_request_->attn_dp_group_id = attn_dp_group_id;

  UpdateBlockPtrs(forward_request_->kv_cache_ptrs);
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
  AppendFlatKVCacheBlkIds(model_instance->GetLayerNum(), kv_cache_blocks, forward_request_->atb_kv_cache_base_blk_ids,
                          cache_manager);
#endif

  return forward_request_.get();
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
void AppendFlatKVCacheBlkIds(const uint32_t layer_num, const std::vector<std::vector<int>> &device_block_ids,
                             std::vector<std::vector<int32_t>> &atb_block_ids,
                             std::shared_ptr<CacheManagerInterface> cache_manager) {
  const size_t rank_num = device_block_ids.size();
  for (size_t rank = 0; rank < rank_num; ++rank) {
    // for dedicate device kv blocks
    const size_t base_id = cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank)->GetBlocksBaseId();
    const auto &device_blocks = device_block_ids[rank];
    auto &atb_blocks = atb_block_ids[rank];
    const size_t exist_blocks = atb_blocks.size();
    atb_blocks.resize(device_blocks.size());
    for (size_t i = exist_blocks; i < device_blocks.size(); ++i) {
      // NOTE(karlluo): only support bfloat16 or float16, so we just dedicate sizeof(float16) here
      atb_blocks[i] = (device_blocks[i] - base_id) * layer_num * 2;
    }
  }
}
#endif
}  // namespace ksana_llm
