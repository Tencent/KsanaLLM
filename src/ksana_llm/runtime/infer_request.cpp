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
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      abort_waiter(request->abort_waiter),
      finished(request->finisheds[index]),
      aborted(request->aborted),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      beam_search_group(request->beam_search_group),
      is_cudagraph_capture_request(request->is_cudagraph_capture_request),
      span_context(request->span_context),
      timestamp_in_ms(request->timestamp_in_ms),
      req_ctx(request->req_ctx),
      req_fsm(request->req_fsm),
      kv_comm_request_id(request->kv_comm_request_id),
      grammar_matcher(request->grammar_matcher),
      kv_comm_group_key(request->kv_comm_group_key) {}

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

}  // namespace ksana_llm
