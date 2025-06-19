/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/rpc/rpc_endpoint.h"
#include "ksana_llm/service/inference_engine.h"

namespace ksana_llm {

// The Inference server.
class InferenceServer {
 public:
  InferenceServer(const std::string &config_file, const EndpointConfig &endpoint_config);
  ~InferenceServer();

  // Start the inference server.
  Status Start();

  // Stop the inference server.
  Status Stop();

  // Handle serving request.
  Status Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Handle serving request, in streaming mode.
  Status HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                         const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                         std::shared_ptr<StreamingIterator> &streaming_iterator);

  // Handle serving forward request.
  Status HandleForward(const std::string &request_bytes,
                       const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                       std::string &response_bytes);

 private:
  // The inference engine.
  std::shared_ptr<InferenceEngine> inference_engine_;

  // The local endpoint of this service.
  std::shared_ptr<LocalEndpoint> local_endpoint_;

  // The optional rpc endpoint of this service.
  std::shared_ptr<RpcEndpoint> rpc_endpoint_;

  // The channel between endpoint and inference engine.
  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue_;
};

}  // namespace ksana_llm
