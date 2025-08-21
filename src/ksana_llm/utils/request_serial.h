/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <string>
#include <vector>

#include "msgpack.hpp"
#include "nlohmann/json.hpp"

namespace ksana_llm {

/**
 * The following struct definitions align with the format of the JSON object used in the forward interface of KsanaLLM.
 *
 * `MSGPACK_DEFINE_MAP` generates functions for packing and unpacking a struct to and from a msgpack object with map
 * type, where each field name is the key, and the corresponding field value is the value.
 */

struct TargetRequestSerial {
  std::string target_name;
  std::vector<int> cutoff_layer;
  std::vector<int> token_id;
  std::vector<std::pair<int, int>> slice_pos;
  std::string token_reduce_mode;

  MSGPACK_DEFINE_MAP(target_name, cutoff_layer, token_id, slice_pos, token_reduce_mode);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(TargetRequestSerial, target_name, cutoff_layer, token_id, slice_pos,
                                              token_reduce_mode);
};

struct EmbeddingSliceSerial {
  std::vector<int> pos;
  std::vector<std::vector<float>> embeddings;

  MSGPACK_DEFINE_MAP(pos, embeddings);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(EmbeddingSliceSerial, pos, embeddings);
};

struct RequestSerial {
  std::string prompt;
  std::vector<int> input_tokens;
  EmbeddingSliceSerial input_refit_embedding;
  std::vector<TargetRequestSerial> request_target;

  MSGPACK_DEFINE_MAP(prompt, input_tokens, input_refit_embedding, request_target);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(RequestSerial, prompt, input_tokens, input_refit_embedding,
                                              request_target);
};

// Forward request interface
struct BatchRequestSerial {
  std::vector<RequestSerial> requests;

  MSGPACK_DEFINE_MAP(requests);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(BatchRequestSerial, requests);
};

struct PythonTensorSerial {
  std::string data;
  std::vector<size_t> shape;
  std::string dtype;

  MSGPACK_DEFINE_MAP(data, shape, dtype);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(PythonTensorSerial, data, shape, dtype);
};

struct TargetResponseSerial {
  std::string target_name;
  PythonTensorSerial tensor;

  MSGPACK_DEFINE_MAP(target_name, tensor);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(TargetResponseSerial, target_name, tensor);
};

struct ResponseSerial {
  std::vector<int> input_token_ids;
  std::vector<TargetResponseSerial> response;

  MSGPACK_DEFINE_MAP(input_token_ids, response);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(ResponseSerial, input_token_ids, response);
};

// Forward response interface
struct BatchResponseSerial {
  std::vector<ResponseSerial> responses;  // the list of response data
  std::string message;                    // the response message
  int code;                               // the response code

  // {responses: responses, message: message, code: code}
  MSGPACK_DEFINE_MAP(responses, message, code);
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(BatchResponseSerial, responses, message, code);
};

}  // namespace ksana_llm
