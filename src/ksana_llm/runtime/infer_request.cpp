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
      timestamp_in_us(request->timestamp_in_us),
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
      << ", kv_cached_token_num:" << kv_cached_token_num << ", prefix_cache_len:" << prefix_cache_len
      << ", input_tokens_size:" << input_tokens.size() << ", output_tokens_size:" << output_tokens.size()
      << ", forwarding_tokens_size:" << forwarding_tokens.size() << ", draft_tokens_size:" << draft_tokens.size()
      << ", accepted_tokens_size:" << accepted_tokens.size() << ", generated_token_size:" << generated_tokens.size()
      << PrintKVBlockIds(print_details) << ", swap_pending:" << swap_pending << ", finished:" << finished
      << ", aborted:" << aborted << ", finish_status:" << finish_status.ToString() << " ) ";
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

const std::vector<int> &InferRequest::GetInflightSequence() const { return forwarding_tokens; }

size_t InferRequest::GetInflightSequenceLen() const { return forwarding_tokens.size(); }

size_t InferRequest::GetInflightQueryLen() const { return forwarding_tokens.size() - kv_cached_token_num; }

size_t InferRequest::GetInflightSamplingTokenNum() const { return sampling_token_num; }

size_t InferRequest::GetPlanningSequenceLen() const {
  if (planning_workload_.prefill_token_num > 0) {
    return planning_workload_.prefill_start_offset + planning_workload_.GetTokenNum();
  }
  // Decoding
  if (inflight_task_.IsEmpty()) {
    return forwarding_tokens.size() + planning_workload_.GetTokenNum();
  } else {
    return forwarding_tokens.size() + inflight_task_estimated_generated_token_num_ +
           inflight_task_estimated_draft_token_num_ + planning_workload_.GetTokenNum();
  }
}

size_t InferRequest::GetPlanningQueryLen() const { return planning_workload_.GetTokenNum(); }

size_t InferRequest::GetPlanningSamplingTokenNum() const { return planning_workload_.sampling_token_num; }

void InferRequest::SetKvCachedTokenNum(size_t num) {
  kv_cached_token_num = num;
  prefix_cache_len = num;
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

void InferRequest::UpdateBlockPtrs(std::vector<std::vector<void *>> &block_ptrs) {
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

ForwardRequest *InferRequest::GetForwardRequest() {
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
  forward_request_->is_cudagraph_capture_request = is_cudagraph_capture_request;

  forward_request_->kv_cache_ptrs.resize(kv_cache_blocks.size());
  forward_request_->atb_kv_cache_base_blk_ids.resize(kv_cache_blocks.size());

  forward_request_->infer_stage = infer_stage;
  forward_request_->step = step;
  forward_request_->kv_cached_token_num = kv_cached_token_num;
  forward_request_->logits_custom_length = logits_custom_length;
  forward_request_->sampling_token_num = sampling_token_num;
  forward_request_->last_step_token_num = last_step_token_num;
  forward_request_->logits_offset = logits_offset;
  forward_request_->is_use_prefix_cache = is_use_prefix_cache;
  forward_request_->prefix_cache_len = prefix_cache_len + flexible_cached_copy_tasks.size();
  forward_request_->attn_dp_group_id = attn_dp_group_id;
  forward_request_->block_checksums = &block_checksums;
  forward_request_->checksummed_block_num = &checksummed_block_num;

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

void InferRequest::ResetPrefillingTokens() {
  infer_stage = InferStage::kContext;
  prefilling_tokens_ = output_tokens;
  kv_cached_token_num = 0;
  step = 0;
  suggested_draft_num = 0;
  prefix_cache_len = 0;
  remaining_workload_.Reset();
  remaining_workload_.prefill_token_num = prefilling_tokens_.size();

  // Assumption: if logits_custom_length > 0, then sampling_token_num = logits_custom_length
  //             because user asked to generate logits for a specific number of tokens
  //             In runtime implementation, sampling_token_num is used to determine the number of logits to generate
  remaining_workload_.sampling_token_num = std::max(kStepGenerateTokenNum, logits_custom_length);

  planning_workload_ = remaining_workload_;
}

void InferRequest::SetInflightTaskGenResultEstimation(size_t generated_token_num, size_t draft_token_num) {
  KLLM_CHECK(!inflight_task_.IsEmpty());
  inflight_task_estimated_generated_token_num_ = generated_token_num;
  inflight_task_estimated_draft_token_num_ = draft_token_num;
}

void InferRequest::SetRemainingWorkload(const ScheduleTaskWorkload &workload) { remaining_workload_ = workload; }

void InferRequest::SetPlanningWorkload(const ScheduleTaskWorkload &workload) { planning_workload_ = workload; }

void InferRequest::SetPlanningTask() {
  KLLM_CHECK(planning_task_.IsEmpty());
  planning_task_.workload = planning_workload_;

  KLLM_CHECK(remaining_workload_.prefill_token_num >= planning_workload_.prefill_token_num);
  remaining_workload_.prefill_token_num -= planning_workload_.prefill_token_num;
  remaining_workload_.prefill_start_offset += planning_workload_.prefill_token_num;
  remaining_workload_.generated_token_num = 0;
  remaining_workload_.draft_token_num = 0;
  KLLM_CHECK((remaining_workload_.prefill_token_num + remaining_workload_.prefill_start_offset) ==
             prefilling_tokens_.size());
  planning_workload_.Reset();
}

void InferRequest::UpdateAfterInflightTaskFinished() {
  output_mutex.lock();
  if (inflight_task_.workload.draft_token_num > 0) {
    // replace draft tokens with accepted tokens.
    forwarding_tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num + accepted_tokens.size());
    output_tokens.insert(output_tokens.end(), forwarding_tokens.end() - accepted_tokens.size(),
                         forwarding_tokens.end());
    // forwarding_tokens.insert(forwarding_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
    // output_tokens.insert(output_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
    accepted_tokens.clear();
  }
  // current token has kv_cache
  kv_cached_token_num = forwarding_tokens.size();

  if (generated_tokens.size() > 0) {
    // append new tokens to output_tokens
    output_tokens.insert(output_tokens.end(), generated_tokens.begin(), generated_tokens.end());
  }
  // GenerationController makes sure eos only appears at the end of accepted_tokens + generated_tokens
  if (std::find(sampling_config.stop_token_ids.begin(), sampling_config.stop_token_ids.end(), output_tokens.back()) !=
      sampling_config.stop_token_ids.end()) {
    KLLM_LOG_DEBUG << "req " << req_id << " finished. output_tokens.size=" << output_tokens.size()
                   << ", eos token=" << output_tokens.back();
    is_eos_generated_ = true;
  }
  output_mutex.unlock();

  // If task is prefill task and not the last step, drop draft_tokens.
  // generated_tokens should be empty.
  if (inflight_task_.workload.prefill_token_num > 0 &&
      (inflight_task_.workload.prefill_token_num + inflight_task_.workload.prefill_start_offset <
       prefilling_tokens_.size())) {
    assert(generated_tokens.size() == 0);
    draft_tokens.clear();
  }
  // generated token and draft token are new workload to be processed
  remaining_workload_.generated_token_num = generated_tokens.size();
  remaining_workload_.draft_token_num = draft_tokens.size();

  // Adjust planning workload for scheduling
  assert(!(remaining_workload_.prefill_token_num > 0 &&
           (remaining_workload_.generated_token_num > 0 || remaining_workload_.draft_token_num > 0)));
  planning_workload_.Reset();
  if (remaining_workload_.prefill_token_num > 0) {
    planning_workload_.prefill_token_num = remaining_workload_.prefill_token_num;
  } else {
    planning_workload_.generated_token_num = generated_tokens.size();
    planning_workload_.draft_token_num = draft_tokens.size();
  }
}

void InferRequest::LaunchPlanningTask() {
  KLLM_CHECK(!planning_task_.IsEmpty());
  KLLM_CHECK(inflight_task_.IsEmpty());
  assert(planning_task_.workload.GetTokenNum() > 0);

  inflight_task_ = planning_task_;
  planning_task_.Reset();
  SetKvCachedTokenNum(forwarding_tokens.size());
  if (inflight_task_.workload.prefill_token_num > 0) {
    size_t forwarded_token_num = forwarding_tokens.size();
    if (forwarded_token_num == inflight_task_.workload.prefill_start_offset) {
      forwarding_tokens.insert(
          forwarding_tokens.end(), prefilling_tokens_.begin() + forwarded_token_num,
          prefilling_tokens_.begin() + forwarded_token_num + inflight_task_.workload.prefill_token_num);
    } else {
      // Prefix cache hit
      forwarding_tokens.clear();
      forwarding_tokens.insert(forwarding_tokens.end(), prefilling_tokens_.begin(),
                               prefilling_tokens_.begin() + inflight_task_.workload.prefill_start_offset +
                                   inflight_task_.workload.prefill_token_num);
      SetKvCachedTokenNum(inflight_task_.workload.prefill_start_offset);
    }

  } else {
    auto merged_draft_tokens = draft_tokens.GetDraftTokens();

    size_t resource_ready_token_num =
        inflight_task_.workload.generated_token_num + inflight_task_.workload.draft_token_num;
    KLLM_CHECK(generated_tokens.size() <= resource_ready_token_num);
    forwarding_tokens.insert(forwarding_tokens.end(), generated_tokens.begin(), generated_tokens.end());

    resource_ready_token_num -= generated_tokens.size();
    size_t draft_token_num = std::min(resource_ready_token_num, merged_draft_tokens.size());
    forwarding_tokens.insert(forwarding_tokens.end(), merged_draft_tokens.begin(),
                             merged_draft_tokens.begin() + draft_token_num);

    // Change inflight_task_ workload to real workload
    inflight_task_.workload.generated_token_num = generated_tokens.size();
    inflight_task_.workload.draft_token_num = draft_token_num;
    inflight_task_.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
    if (draft_token_num < merged_draft_tokens.size()) {
      draft_tokens.TruncDraft(draft_token_num);
    }
  }
  sampling_token_num = inflight_task_.workload.sampling_token_num;
  forwarding_tokens_draft_num = inflight_task_.workload.draft_token_num;

  assert(inflight_task_.workload.GetTokenNum() > 0);
  assert(forwarding_tokens.size() > kv_cached_token_num);
}

std::string InferRequest::ScheduleStateToStr() const {
  std::stringstream ss;
  ss << " schedule_state={ req_id=" << req_id << ", inflight_task_=" << inflight_task_.workload.ToString()
     << ", planning_task_=" << planning_task_.workload.ToString()
     << ", remaining_workload_=" << remaining_workload_.ToString()
     << ", planning_workload_= " << planning_workload_.ToString()
     << ", forwarding_tokens.size=" << forwarding_tokens.size() << ", generated_tokens=" << Vector2Str(generated_tokens)
     << ", accepted_tokens=" << Vector2Str(accepted_tokens)
     << ", draft_tokens=" << Vector2Str(draft_tokens.GetDraftTokens())
     << ", output_tokens.size=" << output_tokens.size() << ", is_eos_generated=" << is_eos_generated_ << " }";
  return ss.str();
}

}  // namespace ksana_llm
