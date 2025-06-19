/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/endpoints/rpc/rpc_endpoint.h"

namespace ksana_llm {

class EndpointFactory {
 public:
  // Create a local endpoint instance via the input config.
  static std::shared_ptr<LocalEndpoint> CreateLocalEndpoint(
      const EndpointConfig &endpoint_config, Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue);

  // Create a rpc endpoint instance via the input config.
  static std::shared_ptr<RpcEndpoint> CreateRpcEndpoint(const EndpointConfig &endpoint_config,
                                                        const std::shared_ptr<LocalEndpoint> &local_endpoint);
};

}  // namespace ksana_llm
