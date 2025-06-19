/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>

#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/request.h"

namespace ksana_llm {

// The information used for sampling.
struct SamplingRequest {
  // The req id of the user's request.
  int64_t req_id;

  // The custom length for the logits output, allowing for a specific size of logits to be generated.
  size_t logits_custom_length = 0;

  // The sampling config.
  SamplingConfig* sampling_config = nullptr;

  // The logits buf and offset.
  std::vector<float*> logits_buf;
  size_t logits_offset;

  std::vector<int>* input_tokens = nullptr;

  std::vector<int>* forwarding_tokens = nullptr;

  // Generated tokens in this sampling.
  std::vector<int>* sampling_result_tokens = nullptr;

  size_t sampling_token_num = 1;

  // The key is the request target, which can only be a predefined set of requestable targets {embedding_lookup,
  // layernorm, transformer, logits}.
  const std::map<std::string, TargetDescribe>* request_target = nullptr;

  // The result of request_target.
  std::map<std::string, PythonTensor>* response = nullptr;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>>* logprobs = nullptr;

  // Beam Search Group
  std::vector<std::shared_ptr<InferRequest>>* req_group = nullptr;

  // Model config
  const ModelConfig* model_config = nullptr;

  // The no_reapete_ngram sampling map
  NgramDict* ngram_dict = nullptr;

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx;
};

}  // namespace ksana_llm
