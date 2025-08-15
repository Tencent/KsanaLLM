/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sys/stat.h>
#include <map>
#include <vector>

#include "ksana_llm/profiler/profiler.h"
#include "ksana_llm/utils/finite_state_machine.h"
#include "ksana_llm/utils/grammar_matcher.h"
#include "ksana_llm/utils/id_generator.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace ksana_llm {

struct SamplingConfig {
  int topk = 1;
  int num_beams = 1;
  int num_return_sequences = 1;

  // The smallest set of most probable tokens with probabilities that add up
  // to or higher than `topp` are considered
  float topp = 1.0f;

  // Modulate the next token probabilities
  float temperature = 1.0f;

  // The parameter for repetition penalty. 1.0 means no penalty
  float repetition_penalty = 1.0f;

  // The parameter for length penalty. 1.0 means no penalty
  float length_penalty = 1.0f;

  // Tokens that stop the generation when they are generated.
  // The returned tokens will contain the stop tokens.
  std::vector<int> stop_token_ids;

  // Whether to ignore any EOS tokens and continue generating
  // tokens after an EOS token is generated.
  bool ignore_eos = false;

  // The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
  int max_new_tokens = 1024;

  int logprobs_num = 0;

  // The parameter for no_repeat_ngram_size.
  int no_repeat_ngram_size = 0;
  int encoder_no_repeat_ngram_size = 0;
  int decoder_no_repeat_ngram_size = 0;

  // In generation phasse, when stop strings are meet, the request will be stopped and truncated
  std::vector<std::string> stop_strings;

  // The JSON string for guided Encoding generation.
  std::string json_schema;
  bool enable_structured_output = false;

  // Check and adjust sampling config arguments.
  Status VerifyArgs();
};

typedef std::tuple<std::vector<int>, std::vector<std::vector<std::pair<int, float>>>, float> OutputTuple;

struct __attribute__((visibility("default"))) EmbeddingSlice {
  // The pos indicates the start position of the embedding to be replaced.
  std::vector<int> pos;

  // embeddings is the embedding value to be used for the replacement, from the request.
  std::vector<std::vector<float>> embeddings;

  // The same as embeddings but is python object
  std::vector<py::object> embedding_tensors;

  // Additional tensors computed on Python side and required by C++ for further processing.
  std::vector<py::object> additional_tensors;
};

enum TokenReduceMode {
  GATHER_ALL,
  GATHER_TOKEN_ID,
  INVALID_TYPE,
};

inline TokenReduceMode GetTokenReduceMode(const std::string& token_reduce_mode_str) {
  if (token_reduce_mode_str == "GATHER_ALL") {
    return TokenReduceMode::GATHER_ALL;
  } else if (token_reduce_mode_str == "GATHER_TOKEN_ID") {
    return TokenReduceMode::GATHER_TOKEN_ID;
  }
  return TokenReduceMode::INVALID_TYPE;
}

struct TargetDescribe {
  // The IDs of special tokens in the request target. Based on these IDs, the corresponding target tensor (hidden state,
  // logits, etc.) should be returned.
  std::vector<int> token_id;
  // The position intervals (inclusive of both ends) of token segments in the request target. The target tensor (hidden
  // state, logits, etc.) should be returned based on the defined intervals.
  std::vector<std::pair<int, int>> slice_pos;
  // The reduction operation mode for each token_id when returning values.
  TokenReduceMode token_reduce_mode;
};

struct KsanaPythonInput {
  // The requested model name.
  std::string model_name;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The tokens of this request.
  std::vector<int> input_tokens;

  // Embedding slice used to refit input embedding
  EmbeddingSlice input_refit_embedding;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::map<std::string, TargetDescribe> request_target;

  // The structured output regex to build the finite state machine.
  std::string structured_output_regex;

  // Verifiy that the above `request_target` describes valid targets.
  // This function also converts negative `slice_pos` parameters to their corresponding positive values, if exist.
  Status VerifyRequestTarget();
};

// In the Python environment, define tensor class.
struct PythonTensor {
  std::vector<uint8_t> data;
  std::vector<size_t> shape;
  std::string dtype;
};

// A vector containing pointers to FlexibleCachedCopyTask objects, which represent tasks that involve copying data
// flexibly between different memory regions.
struct FlexibleCachedCopyTask {
  FlexibleCachedCopyTask() = default;
  void Update(int dst_token_idx, int src_token_idx, std::vector<int>& dst_block_id, std::vector<int>& src_block_id) {
    dst_token_idx_ = dst_token_idx;
    src_token_idx_ = src_token_idx;
    dst_block_id_ = dst_block_id;
    src_block_id_ = src_block_id;
  }
  int dst_token_idx_ = 0;
  int src_token_idx_ = 0;
  std::vector<int> dst_block_id_;
  std::vector<int> src_block_id_;
};

class Request {
 public:
  // Build Request based on the given KsanaPythonInput.
  // The lifetime of the KsanaPythonInput object must be longer than Request,
  // since some members in Request are references to KsanaPythonInput.
  explicit Request(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                   const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx);

  ~Request() { KLLM_LOG_DEBUG << "Request " << req_id << " destroyed"; }

  // The unique id of a request.(Deprecated)
  int64_t req_id;

  // TODO(zakwang): Replace req_id
  std::vector<int64_t> req_ids;

  // The requested model name.
  std::string model_name;

  // The input tokens of this request.
  std::vector<int> input_tokens;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // Embedding slice used to refit input embedding
  EmbeddingSlice& input_refit_embedding;

  // TODO(zakwang): Replace output_tokens
  std::vector<OutputTuple> output_group;

  // The intermediate result of beam_search
  std::vector<OutputTuple> beam_search_group;

  // The output tokens of this request.
  std::vector<int>& output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>>& logprobs;

  // The config of sampling.
  SamplingConfig sampling_config;

  // The waiter notified when request finished.
  std::shared_ptr<Waiter> waiter = nullptr;

  // The waiter notified when step finished.
  std::shared_ptr<Waiter> step_waiter = nullptr;

  // The waiter notified when request abortd.
  std::shared_ptr<Waiter> abort_waiter = nullptr;

  // TODO(zakwang): Replace finished
  std::deque<bool> finisheds;

  // Whether the request is finished.
  bool& finished;

  // Whether the request hve been aborted by client.
  bool aborted = false;

  // The finish status of this request.
  Status finish_status;

  // Protect parallel access for output token.
  std::mutex output_mutex;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe>& request_target;

  // The result of request_target.
  std::map<std::string, PythonTensor> response;

  // Whether this request is the last one in a batch of requests.
  bool last_in_batch = true;

  // is cudagraph capture call
  bool is_cudagraph_capture_request = false;

  // Opentelemetry SpanContext
  opentelemetry::trace::SpanContext span_context = opentelemetry::trace::SpanContext::GetInvalid();

  // The arrive time.
  uint64_t timestamp_in_ms;

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // Whether the request contains stop strings
  bool has_stop_strings = false;

  // The FiniteStateMachine when using structured output optimization.
  std::shared_ptr<FiniteStateMachine> req_fsm;

  // grammar matcher
  std::shared_ptr<GrammarMatcherWrapper> grammar_matcher = nullptr;
  bool grammar_finished = false;

  // This is a unique ID for the KV transformer group.
  int64_t kv_comm_request_id;

  // This key for kv transformer group.
  std::string kv_comm_group_key;

 private:
  // The id generator
  inline static IdGenerator id_generator_;
};

struct KsanaPythonOutput {
  KsanaPythonOutput() = default;

  // Build KsanaPythonOutput based on the specified Request.
  explicit KsanaPythonOutput(std::shared_ptr<Request> req);

  // The input tokens of this request.
  std::vector<int> input_tokens;

  // The output tokens of this request.
  std::vector<std::vector<int>> output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::vector<std::pair<int, float>>>> logprobs;

  // Embedding value for plugin output
  std::vector<std::vector<float>> embedding;

  // The result of request_target.
  std::map<std::string, PythonTensor> response;

  // The finish status of this request.
  Status finish_status;
};

}  // namespace ksana_llm
