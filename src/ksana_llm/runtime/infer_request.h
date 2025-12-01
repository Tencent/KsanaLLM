/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <string>
#include <vector>

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/runtime/draft_generator/draft_tokens.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/runtime/structured_generation/structured_generator_interface.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// Workload of a task
struct ScheduleTaskWorkload {
  size_t prefill_token_num;
  size_t prefill_start_offset;
  size_t generated_token_num;
  size_t draft_token_num;
  size_t sampling_token_num;
  ScheduleTaskWorkload() { Reset(); }
  bool IsEmpty() const { return prefill_token_num == 0 && generated_token_num == 0 && draft_token_num == 0; }
  size_t GetTokenNum() const { return prefill_token_num + generated_token_num + draft_token_num; }
  void Reset() {
    prefill_token_num = 0;
    prefill_start_offset = 0;
    generated_token_num = 0;
    draft_token_num = 0;
    sampling_token_num = 0;
  }

  const std::string ToString() const {
    if (IsEmpty()) {
      return "()";
    }

    std::ostringstream ss;
    ss << "( ";
    if (prefill_token_num > 0) {
      ss << "prefill_token_num=" << prefill_token_num << ", prefill_start_offset=" << prefill_start_offset;
    }
    if (generated_token_num > 0) {
      ss << " generated_token_num=" << generated_token_num;
    }
    if (draft_token_num > 0) {
      ss << " draft_token_num=" << draft_token_num;
    }
    if (sampling_token_num > 0) {
      ss << " sampling_token_num=" << sampling_token_num;
    }
    ss << " )";
    return ss.str();
  }
};

struct ScheduleTask {
  ScheduleTaskWorkload workload;
  bool IsEmpty() const { return workload.IsEmpty(); }
  void Reset() { workload.Reset(); }
};

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest(std::shared_ptr<Request> &request, const int index);
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

  // Update addr ptr of blocks.
  void UpdateBlockPtrs(std::vector<std::vector<void *>> &block_ptrs);

  void RebuildBlockPtrs() { reset_forward_request_ = true; }

  // Get this infer request's KV occupied devices.
  std::vector<int> GetKVOccupiedDevices();

  std::string PrintKVBlockIds(bool print_details = false) const;

  std::string ToString(bool print_details = false) const;

  friend std::ostream &operator<<(std::ostream &os, const InferRequest &req);

 public:
  // In a step, forward() takes sequence and query as input.
  // sequence is tokens can be input tokens + generated tokens + draft tokens).
  // query is tokens processed in this step. It is last part of sequence.
  //      It can be part of input tokens, or generated tokens + draft tokens.

  // Get sequence for inflight step, all tokens have kv cache space
  const std::vector<int> &GetInflightSequence() const;
  // Get sequence length for inflight step
  size_t GetInflightSequenceLen() const;
  // Get query len in inflight step
  size_t GetInflightQueryLen() const;
  // Get the number of tokens to be sampled in inflight step
  size_t GetInflightSamplingTokenNum() const;
  // After inflight step executed, adjust sequence accoding to generation result
  // size_t SetInFlightGenResult(const GenResult& result);

  // Get sequence length for planning step, all tokens need kv cache
  size_t GetPlanningSequenceLen() const;
  // Get query len in planning step
  size_t GetPlanningQueryLen() const;
  // Get/Set the number of tokens to be sampled in planning step
  size_t GetPlanningSamplingTokenNum() const;

  void SetPlanningGeneratedTokenNum(size_t num);
  void SetPlanningDraftTokenNum(size_t num);

  void SetKvCachedTokenNum(size_t num);

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

  // Suggested number of draft tokens to generate, determined by the scheduler
  size_t suggested_draft_num = 0;

  // accepted draft tokens
  std::vector<int> accepted_tokens;

  // draft token num in forwarding_tokens
  size_t forwarding_tokens_draft_num = 0;

  // last step draft num, used in async schedule
  size_t last_step_draft_num = 0;

  // token generated by model, complete new tokens in a step are (draft_tokens - reject_token_num) + generated_tokens
  std::vector<int> generated_tokens;

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

  StructuredGeneratorConfig &structured_generator_config;

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

  // Whether the req is mock.
  bool is_mock_req = 0;

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
  int64_t kv_comm_request_id = 0;

  // This key for kv transformer group.
  std::string kv_comm_group_key;

  // structured generator
  std::shared_ptr<StructuredGeneratorInterface> structured_generator = nullptr;

  /*******************************************************************
   * State info used in generation
   * TODO (robertyuan): Move them into a structure later
   *******************************************************************/
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
  size_t kv_cached_token_num = 0;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The max token number of one block.
  size_t block_token_num;

  // Checksum for every block on every rank.
  std::vector<std::vector<size_t>> block_checksums;

  // The number of blocks that have been checksummed on each rank.
  std::vector<size_t> checksummed_block_num;

  // The offset for model forward's logits output.
  size_t logits_offset = 0;

  // Whether the current req is in pending status of swappiness.
  bool swap_pending = false;

  // The swappiness future.
  std::future<void> swap_future;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
  // flexibly between different memory regions.
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

  // The no_repeate ngram sampling map
  NgramDict ngram_dict;

  // is cudagraph capture call
  bool &is_cudagraph_capture_request;

  // The arrive time.
  uint64_t timestamp_in_us;

  // request context
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // Incremental decoded str used in stop strings
  std::string incremental_decoded_str;

  // The number of tokens that have been computed.
  size_t computed_token_num = 0;

 public:
  // ForwardRequest's lifecycle is bound to InferRequest's, making smart pointers redundant. Any use of ForwardRequest*
  // requires the guaranteed existence of its associated InferRequest.
  ForwardRequest *GetForwardRequest();
  SamplingRequest *GetSamplingRequest(const size_t multi_batch_id);
  std::unique_ptr<ForwardRequest> forward_request_;
  std::unique_ptr<SamplingRequest> sampling_request_;
  bool reset_forward_request_ = true;

 public:
  // Init or Recompute, copy output_tokens to prefilling_tokens_
  void ResetPrefillingTokens();

  const std::vector<int> &GetPrefillingTokens() const { return prefilling_tokens_; }
  const ScheduleTask &GetInflightTask() const { return inflight_task_; }
  const ScheduleTaskWorkload &GetRemainingWorkload() const { return remaining_workload_; }
  const ScheduleTaskWorkload &GetPlanningWorkload() const { return planning_workload_; }

  bool HasInflightTask() const { return !inflight_task_.IsEmpty(); }
  bool HasPlanningTask() const { return !planning_task_.IsEmpty(); }
  void SetInflightTaskGenResultEstimation(size_t generated_token_num, size_t draft_token_num);

  void SetRemainingWorkload(const ScheduleTaskWorkload &workload);
  void SetPlanningWorkload(const ScheduleTaskWorkload &workload);
  void SetPlanningTask();

  void ResetInflightTask() { inflight_task_.Reset(); }

  void UpdateAfterInflightTaskFinished();

  bool IsEosGenerated() const { return is_eos_generated_; }

  void LaunchPlanningTask();

  std::string ScheduleStateToStr() const;

  bool IsStopped() const { return is_stopped_; }
  void Stop() { is_stopped_ = true; }

 private:
  bool is_stopped_ = false;
  std::vector<int> prefilling_tokens_;
  ScheduleTask inflight_task_;
  size_t inflight_task_estimated_generated_token_num_ = 0;
  size_t inflight_task_estimated_draft_token_num_ = 0;

  ScheduleTask planning_task_;
  ScheduleTaskWorkload planning_workload_;
  ScheduleTaskWorkload remaining_workload_;
  bool is_eos_generated_ = false;
};

#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
void AppendFlatKVCacheBlkIds(const uint32_t layer_num, const std::vector<std::vector<int>> &device_block_ids,
                             std::vector<std::vector<int32_t>> &atb_block_ids,
                             std::shared_ptr<CacheManagerInterface> cache_manager);
#endif

}  // namespace ksana_llm
