/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_server.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

InferenceServer::InferenceServer(const std::string &config_file, const EndpointConfig &endpoint_config) {
  InitLoguru();
  KLLM_LOG_INFO << "Log levels: " << Vector2Str(GetLogLevels());

  KLLM_LOG_INFO << "Init inference server with config file: " << config_file;
  auto env = Singleton<Environment>::GetInstance();
  if (!env) {
    KLLM_THROW("The Environment is nullptr.");
  }
  STATUS_CHECK_FAILURE(env->ParseConfig(config_file));

  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);
  pipeline_config.SetDistributeRelatedConfig();
  env->SetPipelineConfig(pipeline_config);

  Singleton<Environment>::GetInstance()->InitializeExpertParallelConfig();

  // Init inference engine.
  inference_engine_ = std::make_shared<InferenceEngine>(request_queue_);

  // Init local endpoint.
  local_endpoint_ = EndpointFactory::CreateLocalEndpoint(endpoint_config, request_queue_);
  // Init rpc endpoint if specified.
  if (endpoint_config.type == EndpointType::RPC) {
    rpc_endpoint_ = EndpointFactory::CreateRpcEndpoint(endpoint_config, local_endpoint_);
  }

  KLLM_LOG_INFO << "Inference server is initialized.";
}

InferenceServer::~InferenceServer() { KLLM_LOG_DEBUG << "InferenceServer destroyed."; }

Status InferenceServer::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                               const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                               ksana_llm::KsanaPythonOutput &ksana_python_output) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->Handle(ksana_python_input, req_ctx, ksana_python_output));
}

Status InferenceServer::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                        const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                        std::shared_ptr<StreamingIterator> &streaming_iterator) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleStreaming(ksana_python_input, req_ctx, streaming_iterator));
}

Status InferenceServer::HandleForward(const std::string &request_bytes,
                                      const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                      std::string &response_bytes) {
  STATUS_CHECK_AND_REPORT(local_endpoint_->HandleForward(request_bytes, req_ctx, response_bytes));
}

Status InferenceServer::Start() {
  KLLM_LOG_INFO << "Inference server start.";

  inference_engine_->Start();
  if (rpc_endpoint_) {
    rpc_endpoint_->Start();
  }

  return Status();
}

Status InferenceServer::Stop() {
  KLLM_LOG_INFO << "Inference server stop.";

  inference_engine_->Stop();
  if (rpc_endpoint_) {
    rpc_endpoint_->Stop();
  }

  // Force exit here.
  KLLM_LOG_INFO << "Inference server exit.";
  return Status();
}

}  // namespace ksana_llm
