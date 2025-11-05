/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/script.h>
#include <memory>
#include <string>

#include "ksana_llm/service/inference_server.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The torch OP for inference.
class ServingOp : public torch::jit::CustomClassHolder {
 public:
  ServingOp();
  ~ServingOp();

  // Initialize the serving server.
  // think_end_token_id: Optional token ID for reasoning mode. -1 means disabled.
  void InitServing(const std::string &config_file, int think_end_token_id = -1);

  // Generate a response.
  Status Generate(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                  const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                  ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Generate a response, in streaming mode.
  Status GenerateStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                           const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                           std::shared_ptr<StreamingIterator> &streaming_iterator);

  /**
   * Generate responses for requests toward the forward interface.
   * This interface performs a single-step inference, i.e., the context phase.
   * It allows a batch of requests to be submitted at once, and each request
   * can specify multiple targets to obtain model outputs such as transformer,
   * final layer norm, logits, etc.
   * Refer to serving_forward_client for more details.
   */
  Status Forward(const std::string &request_bytes,
                 const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                 std::string &response_bytes);

 public:
  // The config of endpoint.
  EndpointConfig endpoint_config_;

 private:
  // The inference server.
  std::shared_ptr<InferenceServer> inference_server_;
};

}  // namespace ksana_llm
