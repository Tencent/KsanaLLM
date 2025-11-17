/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "ksana_llm/utils/request_packer.h"

#include "base64.hpp"
#include "msgpack.hpp"
#include "nlohmann/json.hpp"

#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

Status RequestPacker::Unpack(const std::string& request_bytes,
                             std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                             const std::optional<std::string>& content_type) {
  if (content_type.has_value()) {
    if (content_type.value() == "application/json") {
      return UnpackJson(request_bytes, ksana_python_inputs);
    } else if (content_type.value() == "application/x-msgpack") {
      return UnpackMsgpack(request_bytes, ksana_python_inputs);
    } else {
      return Status(RET_INVALID_ARGUMENT, "The Unpack content_type is not supported.");
    }
  } else {
    // default to msgpack.
    return UnpackMsgpack(request_bytes, ksana_python_inputs);
  }
}

Status RequestPacker::Pack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                           const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
                           std::string& response_bytes, const std::optional<std::string>& content_type) {
  if (content_type.has_value()) {
    if (content_type.value() == "application/json") {
      return PackJson(ksana_python_inputs, ksana_python_outputs, response_status, response_bytes);
    } else if (content_type.value() == "application/x-msgpack") {
      return PackMsgpack(ksana_python_inputs, ksana_python_outputs, response_status, response_bytes);
    } else {
      return Status(RET_INVALID_ARGUMENT, "The Pack content_type is not supported.");
    }
  } else {
    return Status(RET_INVALID_ARGUMENT, "The Pack content_type is empty.");
  }
}

KsanaPythonInput RequestPacker::GetKsanaPythonInput(const RequestSerial& req) {
  KsanaPythonInput ksana_python_input;
  if (req.input_tokens.empty()) {  // If input tokens are empty, tokenize the prompt.
    Singleton<Tokenizer>::GetInstance()->Encode(req.prompt, ksana_python_input.input_tokens);
  } else {
    ksana_python_input.input_tokens = req.input_tokens;
  }
  ksana_python_input.input_refit_embedding.pos = req.input_refit_embedding.pos;
  ksana_python_input.input_refit_embedding.embeddings = req.input_refit_embedding.embeddings;
  for (const auto& [target_name, cutoff_layer, token_id, slice_pos, token_reduce_mode] : req.request_target) {
    ksana_python_input.request_target.emplace(
        target_name, TargetDescribe{cutoff_layer, token_id, slice_pos, GetTokenReduceMode(token_reduce_mode)});
  }
  // Verify the request target and throw an exception if anything is invalid.
  ksana_python_input.VerifyRequestTarget();
  // For forward interface.
  ksana_python_input.sampling_config.max_new_tokens = 1;
  return ksana_python_input;
}

ResponseSerial RequestPacker::GetResponseSerial(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                                                const KsanaPythonOutput& ksana_python_output) {
  ResponseSerial rsp;
  rsp.input_token_ids = ksana_python_input->input_tokens;
  for (const auto& [target_name, tensor] : ksana_python_output.response) {
    rsp.response.push_back(TargetResponseSerial{
        target_name, PythonTensorSerial{base64::encode_into<std::string>(tensor.data.begin(), tensor.data.end()),
                                        tensor.shape, tensor.dtype}});
  }
  return rsp;
}

void RequestPacker::GetKsanaPythonInputs(const BatchRequestSerial& batch_req,
                                         std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs) {
  ksana_python_inputs.clear();
  ksana_python_inputs.reserve(batch_req.requests.size());
  for (const auto& req : batch_req.requests) {
    ksana_python_inputs.push_back(std::make_shared<KsanaPythonInput>(GetKsanaPythonInput(req)));
  }
}

void RequestPacker::GetResponseSerials(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                                       const std::vector<KsanaPythonOutput>& ksana_python_outputs,
                                       const Status& response_status, BatchResponseSerial& batch_rsp) {
  const size_t batch_size =
      response_status.OK()
          ? ksana_python_outputs.size()
          : 0ul;  // If the request failed, do not return the responses, and only return the message and code.
  batch_rsp.responses.clear();
  batch_rsp.responses.reserve(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    batch_rsp.responses.push_back(GetResponseSerial(ksana_python_inputs[i], ksana_python_outputs[i]));
  }
  batch_rsp.message = response_status.GetMessage();
  batch_rsp.code = static_cast<int>(response_status.GetCode());
}

Status RequestPacker::UnpackMsgpack(const std::string& request_bytes,
                                    std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs) {
  // Return early if `request_bytes` is empty.
  if (request_bytes.empty()) {
    return Status(RET_INVALID_ARGUMENT, "Request content is empty.");
  }

  // Construct a KsanaPythonInput object from a RequestSerial object.
  [[maybe_unused]] auto GetKsanaPythonInput = [this](const RequestSerial& req) -> KsanaPythonInput {
    KsanaPythonInput ksana_python_input;
    if (req.input_tokens.empty()) {  // If input tokens are empty, tokenize the prompt.
      Singleton<Tokenizer>::GetInstance()->Encode(req.prompt, ksana_python_input.input_tokens);
    } else {
      ksana_python_input.input_tokens = req.input_tokens;
    }
    ksana_python_input.input_refit_embedding.pos = req.input_refit_embedding.pos;
    ksana_python_input.input_refit_embedding.embeddings = req.input_refit_embedding.embeddings;
    for (const auto& [target_name, cutoff_layer, token_id, slice_pos, token_reduce_mode] : req.request_target) {
      ksana_python_input.request_target.emplace(
          target_name, TargetDescribe{cutoff_layer, token_id, slice_pos, GetTokenReduceMode(token_reduce_mode)});
    }
    // Verify the request target and throw an exception if anything is invalid.
    ksana_python_input.VerifyRequestTarget();
    // For forward interface.
    ksana_python_input.sampling_config.max_new_tokens = 1;
    return ksana_python_input;
  };

  // Try unpack the request bytes and parse into a batch of KsanaPythonInput objects.
  try {
    auto handle = msgpack::unpack(request_bytes.data(), request_bytes.size());
    auto object = handle.get();
    auto batch_req = object.as<BatchRequestSerial>();
    GetKsanaPythonInputs(batch_req, ksana_python_inputs);
    return Status();
  } catch (const msgpack::unpack_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to unpack the request bytes.");
  } catch (const msgpack::type_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to parse the request.");
  } catch (const py::error_already_set& e) {
    PyErr_Clear();
    return Status(RET_INVALID_ARGUMENT, "Failed to decode the input prompt.");
  } catch (const std::runtime_error& e) {  // The request specifies invalid target.
    return Status(RET_INVALID_ARGUMENT, e.what());
  } catch (...) {
    return Status(RET_INVALID_ARGUMENT, "Unknown error occurred during request msgpack unpack.");
  }
}

Status RequestPacker::UnpackJson(const std::string& request_bytes,
                                 std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs) {
  if (request_bytes.empty()) {
    return Status(RET_INVALID_ARGUMENT, "Json Request content is empty.");
  }

  // Try unpack the request bytes and parse into a batch of KsanaPythonInput objects.
  try {
    auto handle = nlohmann::json::parse(request_bytes);
    BatchRequestSerial batch_req = handle.get<BatchRequestSerial>();  // Use get<T>() method
    KLLM_LOG_INFO << "Start GetKsanaPythonInputs from  json request body.";
    GetKsanaPythonInputs(batch_req, ksana_python_inputs);
    KLLM_LOG_INFO << "End GetKsanaPythonInputs from  json request body.";
    return Status();
  } catch (const nlohmann::json::parse_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to unpack the request bytes.");
  } catch (const nlohmann::json::type_error& e) {
    return Status(RET_INVALID_ARGUMENT, "Failed to parse the request.");
  } catch (const py::error_already_set& e) {
    PyErr_Clear();
    return Status(RET_INVALID_ARGUMENT, "Failed to decode the input prompt.");
  } catch (const std::runtime_error& e) {  // The request specifies invalid target.
    return Status(RET_INVALID_ARGUMENT, e.what());
  } catch (...) {
    KLLM_LOG_ERROR << "Unknown exception during JSON unpack. Request bytes length: " << request_bytes.length();
    KLLM_LOG_ERROR << "Request bytes (first 500 chars): " << request_bytes.substr(0, 500);
    return Status(RET_INVALID_ARGUMENT, "Unknown error occurred during request json unpack.");
  }
}

Status RequestPacker::PackJson(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                               const std::vector<KsanaPythonOutput>& ksana_python_outputs,
                               const Status& response_status, std::string& response_bytes) {
  BatchResponseSerial batch_rsp;
  GetResponseSerials(ksana_python_inputs, ksana_python_outputs, response_status, batch_rsp);

  nlohmann::json json_rsp = batch_rsp;
  response_bytes = json_rsp.dump();

  return Status();
}

Status RequestPacker::PackMsgpack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                                  const std::vector<KsanaPythonOutput>& ksana_python_outputs,
                                  const Status& response_status, std::string& response_bytes) {
  // Convert the batch of KsanaPythonOutput objects into BatchResponseSerial objects and pack to response bytes.
  msgpack::sbuffer sbuf;
  BatchResponseSerial batch_rsp;
  GetResponseSerials(ksana_python_inputs, ksana_python_outputs, response_status, batch_rsp);
  msgpack::pack(sbuf, batch_rsp);

  response_bytes.assign(sbuf.data(), sbuf.size());
  return Status();
}

}  // namespace ksana_llm
