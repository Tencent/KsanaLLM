/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/grammar_matcher.h"
#include "ksana_llm/utils/request.h"

namespace ksana_llm {

// The information used for sampling.
struct SamplingRequest {
  // The req id of the user's request.
  int64_t req_id;

  int step = 0;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The sampling config.
  SamplingConfig* sampling_config = nullptr;

  // The logits buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  std::shared_ptr<std::vector<int>> input_tokens;

  std::vector<int>* forwarding_tokens = nullptr;

  // NOTE(david): should be modified later, only deepcopy infer_request
  std::vector<int>* origin_tokens = nullptr;

  bool is_deepcopy = false;

  // Generated tokens in this sampling.
  std::vector<int>* sampling_result_tokens = nullptr;

  size_t sampling_token_num = 1;

  size_t last_step_token_num = 1;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  std::shared_ptr<const std::map<std::string, TargetDescribe>> request_target;

  // The result of request_target.
  std::map<std::string, PythonTensor>* response = nullptr;

  // Store token and their corresponding float probability values.
  std::shared_ptr<std::vector<std::vector<std::pair<int, float>>>> logprobs;

  // Beam Search Group
  std::vector<std::shared_ptr<InferRequest>>* req_group = nullptr;

  // Model config
  const ModelConfig* model_config = nullptr;

  // The no_reapete_ngram sampling map
  NgramDict* ngram_dict = nullptr;

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;

  // Grammar matcher for constrained generation, defaults to nullptr (no grammar constraints)
  std::shared_ptr<GrammarMatcherWrapper> grammar_matcher;

  // Flag to control whether to apply grammar constraints during sampling
  // Used in MTP mode to disable grammar for draft token generation
  bool apply_grammar_constraint = true;
};

}  // namespace ksana_llm
