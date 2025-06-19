/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/endpoints/streaming/streaming_iterator.h"
#include "ksana_llm/utils/channel.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/request_packer.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class LocalEndpoint {
 public:
  LocalEndpoint(const EndpointConfig& endpoint_config,
                Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue);

  // Handle a request.
  virtual Status Handle(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                        const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                        ksana_llm::KsanaPythonOutput& ksana_python_output);

  // Handle a streaming request.
  virtual Status HandleStreaming(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                                 const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                                 std::shared_ptr<StreamingIterator>& streaming_iterator);

  // Handle a forward request.
  virtual Status HandleForward(const std::string& request_bytes,
                               const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx,
                               std::string& response_bytes);

 private:
  // The endpoint config.
  EndpointConfig endpoint_config_;

  // The channel used to pass request from endpoint to inference engine.
  Channel<std::pair<Status, std::shared_ptr<Request>>>& request_queue_;

  // Used in the forward interface to unpack requests and pack responses.
  RequestPacker request_packer_;
};

}  // namespace ksana_llm
