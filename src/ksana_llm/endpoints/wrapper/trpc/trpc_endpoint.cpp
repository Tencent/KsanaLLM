/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/wrapper/trpc/trpc_endpoint.h"

#include "ksana_llm/endpoints/local/local_endpoint.h"
#include "ksana_llm/utils/optional_file.h"
#include "trpc/log/trpc_log.h"
#include "trpc/robust/trpc_robust_api.h"
#include "trpc/server/http_service.h"
#include "trpc/util/http/routes.h"

namespace ksana_llm {

CommonHandler::CommonHandler(const EndpointConfig &endpoint_config,
                             const std::shared_ptr<LocalEndpoint> &local_endpoint)
    : endpoint_config_(endpoint_config), local_endpoint_(local_endpoint) {}

std::shared_ptr<std::unordered_map<std::string, std::string>> CommonHandler::GetTraceContext(
    const trpc::http::HttpRequestPtr &req) const {
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  for (const std::string &field : fields_to_extract_) {
    const std::string field_value = req->GetHeader(field);
    // If field_value is empty, it means the key field does not exist.
    if (!field_value.empty()) {
      req_ctx->emplace(field, std::move(field_value));
    }
  }
  return req_ctx;
}

ForwardHandler::ForwardHandler(const EndpointConfig &endpoint_config,
                               const std::shared_ptr<LocalEndpoint> &local_endpoint)
    : CommonHandler(endpoint_config, local_endpoint) {}

void ForwardHandler::Post(const trpc::ServerContextPtr &ctx, const trpc::http::HttpRequestPtr &req,
                          trpc::http::HttpResponse *rsp) {
  const std::string &request_bytes = req->GetContent();
  // Get trace context from the request header.
  auto req_ctx = GetTraceContext(req);
  std::string response_bytes;

  // Handle the forward request on the local endpoint.
  Status status = local_endpoint_->HandleForward(request_bytes, req_ctx, response_bytes);
  rsp->SetMimeType("application/x-msgpack");
  rsp->SetContent(response_bytes);

  // Request failed.
  if (!status.OK()) {
    rsp->SetStatus(trpc::http::HttpResponse::StatusCode::kInternalServerError);
    if (endpoint_config_.access_log) {
      const std::string status_code = rsp->StatusCodeToString();
      TRPC_LOG_ERROR(fmt::format(
          "{}:{} - \"{} {} HTTP/{}\"{}", ctx->GetIp(), ctx->GetPort(), req->GetMethod(), req->GetUrl(),
          req->GetVersion(), status_code.substr(0, status_code.size() - 2)));  // Ignore the last "\r\n" in status_code.
    }
    return;
  }

  // Request finished.
  rsp->SetStatus(trpc::http::HttpResponse::StatusCode::kOk);
  if (endpoint_config_.access_log) {
    const std::string status_code = rsp->StatusCodeToString();
    TRPC_LOG_INFO(fmt::format(
        "{}:{} - \"{} {} HTTP/{}\"{}", ctx->GetIp(), ctx->GetPort(), req->GetMethod(), req->GetUrl(), req->GetVersion(),
        status_code.substr(0, status_code.size() - 2)));  // Ignore the last "\r\n" in status_code.
  }
}

TrpcEndpoint::TrpcEndpoint(const EndpointConfig &endpoint_config, const std::shared_ptr<LocalEndpoint> &local_endpoint)
    : RpcEndpoint(endpoint_config, local_endpoint) {}

Status TrpcEndpoint::Start() {
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  // Search the tRPC config file within the directory of this file and the rpc_config folder in the ksana_llm Python
  // package.
  const std::string &config_file = optional_file->GetOptionalFile(
      std::filesystem::path(__FILE__).parent_path().string() + "/rpc_config", "rpc_config", "trpc_ksana.yaml");
  // Parse the given tRPC config file.
  if (trpc::TrpcConfig::GetInstance()->Init(config_file) != 0) {
    std::cerr << fmt::format("Load Trpc endpoint config {} failed.", config_file) << std::endl;
    std::abort();
  }
  // Update the tRPC service config.
  auto &service_config = trpc::TrpcConfig::GetInstance()->GetMutableServerConfig().services_config.front();
  // Set ip and port based on the input endpoint config.
  service_config.ip = endpoint_config_.host;
  service_config.port = endpoint_config_.port;
  // Verify the validity of the config.
  if (!trpc::TrpcConfig::GetInstance()->IsValid()) {
    std::cerr << fmt::format("The Trpc endpoint config {} is invalid.", config_file) << std::endl;
    std::abort();
  }

  // Record the service name.
  service_name_ = service_config.service_name;

  // Initialize tRPC logging.
  if (trpc::TrpcPlugin::GetInstance()->InitLogging() != 0) {
    std::cerr << "Initialize Trpc logging failed." << std::endl;
    std::abort();
  }

  // Start the tRPC server.
  trpc_server_thread_ = std::thread([&]() {
    TRPC_LOG_INFO("Startup Trpc endpoint...");
    InitializeRuntime();
  });
  return Status();
}

Status TrpcEndpoint::Stop() {
  TRPC_LOG_INFO("Shutdown Trpc endpoint...");

  DestroyRuntime();
  trpc_server_thread_.join();
  return Status();
}

int TrpcEndpoint::Initialize() {
  auto http_service = std::make_shared<trpc::HttpService>();
  trpc::robust::Start();

  // Define the forward interface.
  auto SetHttpRoutes = [this](trpc::http::HttpRoutes &routes) {
    auto forward_handler = std::make_shared<ForwardHandler>(endpoint_config_, local_endpoint_);
    routes.Add(trpc::http::OperationType::POST, trpc::http::Path("/forward"), forward_handler);
  };
  http_service->SetRoutes(SetHttpRoutes);

  RegisterService(service_name_, http_service);
  return 0;
}

int TrpcEndpoint::RegistryPlugins() {
  // Register the tRPC log plugin.
  trpc::robust::Init();
  return 0;
}

void TrpcEndpoint::Destroy() {
  trpc::log::Destroy();
  trpc::robust::Terminate();
}

}  // namespace ksana_llm
