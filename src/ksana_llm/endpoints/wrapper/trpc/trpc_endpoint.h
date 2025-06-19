/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ksana_llm/endpoints/rpc/rpc_endpoint.h"
#include "trpc/common/trpc_app.h"
#include "trpc/util/http/http_handler.h"

namespace ksana_llm {

// Common parts of all handlers.
class CommonHandler {
 protected:
  CommonHandler(const EndpointConfig& endpoint_config, const std::shared_ptr<LocalEndpoint>& local_endpoint);

  // Extract trace context from http header of the request.
  std::shared_ptr<std::unordered_map<std::string, std::string>> GetTraceContext(
      const trpc::http::HttpRequestPtr& req) const;

  // The endpoint config.
  EndpointConfig endpoint_config_;

  // Requests are handled by the local endpoint.
  std::shared_ptr<LocalEndpoint> local_endpoint_;

 private:
  // Fields in HTTP header that need to be extracted.
  inline static const std::vector<std::string> fields_to_extract_ = {"x-remote-ip", "traceparent"};
};

// Handler for forward interface.
class ForwardHandler : public CommonHandler, public trpc::http::HttpHandler {
 public:
  ForwardHandler(const EndpointConfig& endpoint_config, const std::shared_ptr<LocalEndpoint>& local_endpoint);

  // Handle requests posted to the forward interface.
  void Post(const trpc::ServerContextPtr& ctx, const trpc::http::HttpRequestPtr& req,
            trpc::http::HttpResponse* rsp) override;
};

class TrpcEndpoint : public RpcEndpoint, public trpc::TrpcApp {
 public:
  TrpcEndpoint(const EndpointConfig& endpoint_config, const std::shared_ptr<LocalEndpoint>& local_endpoint);

  ~TrpcEndpoint() override {}

  // Start the trpc endpoint.
  Status Start() override;

  // Stop the trpc endpoint.
  Status Stop() override;

  // Initialize the trpc app.
  int Initialize() override;

  // Register the trpc plugin.
  int RegistryPlugins() override;

  // Destroy the trpc app.
  void Destroy() override;

 private:
  // The name of this trpc service recorded in the configuration.
  std::string service_name_;

  // The thread running this trpc service.
  std::thread trpc_server_thread_;
};

// Functions exported via shared library.
#ifdef __cplusplus
extern "C" {
#endif

std::shared_ptr<TrpcEndpoint> CreateTrpcEndpoint(const EndpointConfig& endpoint_config,
                                                 const std::shared_ptr<LocalEndpoint>& local_endpoint) {
  return std::make_shared<TrpcEndpoint>(endpoint_config, local_endpoint);
}

#ifdef __cplusplus
}
#endif

}  // namespace ksana_llm
