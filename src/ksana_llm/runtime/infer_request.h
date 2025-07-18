/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/draft_generator/draft_tokens.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest(std::shared_ptr<Request> &request, int index);
  ~InferRequest();

  void SetReqGroup(const std::vector<std::shared_ptr<InferRequest>> &beam_search_infer_group) {
    req_group = beam_search_infer_group;
  }

  // Clear the group of requests.
  void ClearReqGroup() { req_group.clear(); }

  // Notify after request finished.
  void Notify();

  // Notify after step finished.
  void NotifyStep();

  // Get addr ptr of blocks.
  std::vector<std::vector<void *>> GetBlockPtrs();

  std::vector<int> GetVerifiedTokens();

  std::string PrintKVBlockIds(bool print_details = false) const;

  std::string ToString(bool print_details = false) const;

  friend std::ostream &operator<<(std::ostream &os, const InferRequest &req);

 public:
  // The req id of the user's request.
  int64_t req_id;

  // The name of model instance.
  std::string &model_name;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  size_t sampling_token_num = kStepGenerateTokenNum;

  // Record the number of tokens sampled in the previous step.
  size_t last_step_token_num = sampling_token_num;

  // The input tokens.
  std::vector<int> &input_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice &input_refit_embedding;

  // The offset for multimodal rotary position embedding, computed in prefill phase by Python plugin,
  // and used in decode phase.
  int64_t mrotary_embedding_pos_offset = 0;

  // output_tokens is used during computation. When split fuse is enabled, output_tokens contains only the split
  // part. This variable is dynamically updated based on the current computation phase and may not always represent the
  // complete output.
  std::vector<int> &output_tokens;

  // draft token generated by MTP and Trie
  DraftTokens draft_tokens;

  // save the accepted hidden states for mtp input
  Tensor accepted_hidden_states;

  // Suggested number of draft tokens to generate, determined by the scheduler
  size_t suggested_draft_num = 0;

  // accepted draft tokens
  std::vector<int> accepted_tokens;

  // draft token num in forwarding_tokens
  size_t forwarding_tokens_draft_num = 0;

  // token generated by model, complete new tokens in a step are (draft_tokens - reject_token_num) + generated_token
  int generated_token;  // only has kModelGenerateTokenNum(1) token now, use vector while kModelGenerateTokenNum>1

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>> &logprobs;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe> &request_target;

  // The result of request_target.
  std::map<std::string, PythonTensor> &response;

  float cumulative_score;

  // The sampling config of this request.
  SamplingConfig &sampling_config;

  // The waiter used to notify when request finished.
  std::shared_ptr<Waiter> &waiter;

  // The waiter used to notify when step finished.
  std::shared_ptr<Waiter> &step_waiter;

  // The waiter used to notify when request aborted..
  std::shared_ptr<Waiter> &abort_waiter;

  // Whether the request is finished.
  bool &finished;

  // whether the request is aborted.
  bool &aborted;

  // The final status of this request.
  Status &finish_status;

  // Protect parallel access for output token.
  std::mutex &output_mutex;

  std::vector<std::shared_ptr<InferRequest>> req_group;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // Different reqs may have different cache managers.
  std::shared_ptr<CacheManagerInterface> cache_manager;

  // data parallel id of this request.
  uint32_t attn_dp_group_id = 0;

  // This is a unique ID for the KV transformer group.
  int64_t kv_comm_request_id;

  // This key for kv transformer group.
  std::string kv_comm_group_key;

  /*******************************************************************
   * State info used in generation
   * TODO (robertyuan): Move them into a structure later
   *******************************************************************/
 public:
  // forwarding_tokens contains tokens used in forwarding step. There are two parts:
  // 1. tokens have kv-caches, kv_cached_token_num is the number
  // 2. tokens need to be processed, their kv-caches are generated during computation
  std::vector<int> forwarding_tokens;

  // tokens generated in current step
  std::vector<int> sampling_result_tokens;

  // The intermediate result of beam_search
  std::vector<OutputTuple> &beam_search_group;

  // context decode or decode stage.
  InferStage infer_stage;

  // The decode step, 0 for context decode, and then 1, 2, 3...
  int step = 0;

  // The number of tokens for which kv caches have been generated.
  int kv_cached_token_num = 0;

  size_t mtp_kv_cached_token_num = 0;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The max token number of one block.
  size_t block_token_num;

  // The offset for model forward's logits output.
  size_t logits_offset = 0;

  // Whether the current req is in pending status of swappiness.
  bool swap_pending = false;

  // The swappiness future.
  std::future<void> swap_future;

  // The flag for tagging request prefix cache usage
  bool is_use_prefix_cache = false;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
  // flexibly between different memory regions.
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

  // The no_repeate ngram sampling map
  NgramDict ngram_dict;

  // is cudagraph capture call
  bool &is_cudagraph_capture_request;

  // Opentelemetry SpanContext
  opentelemetry::trace::SpanContext span_context;

  // The arrive time.
  uint64_t timestamp_in_ms;

  // request context
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // Incremental decoded str used in stop strings
  std::string incremental_decoded_str;

  // The output structure fsm
  std::shared_ptr<FiniteStateMachine> req_fsm;

  // The current state id of the FSM (Finite State Machine).
  size_t fsm_state_id = 0;
};

}  // namespace ksana_llm
