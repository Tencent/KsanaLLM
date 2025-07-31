/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "resolved_endpoint.h"
#include "polaris/consumer.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {
std::string ResolvedEndpoint::GetAddrWithPolaris(const std::string& endpoint) {
  // 解析 endpoint_，获得 namespace_name 和 service_name
  std::string namespace_name = "Production";
  std::string service_name;
  size_t pos = endpoint.find('/');
  if (pos != std::string::npos) {
    namespace_name = endpoint.substr(0, pos);
    service_name = endpoint.substr(pos + 1);
  } else {
    service_name = endpoint;
  }

  polaris::ServiceKey service_key = {namespace_name, service_name};

  polaris::GetOneInstanceRequest request(service_key);
  polaris::Instance instance;
  polaris::ReturnCode ret;

  // 一、RPC调用前：调用北极星接口获取一个被调服务实例，会执行服务路由和负载均衡
  if ((ret = polaris::Singleton::GetConsumer().GetOneInstance(request, instance)) != polaris::kReturnOk) {
    KLLM_LOG_ERROR << "get instance for service with error:";
    return "";
  }
  // 拼接 host:port 字符串
  return instance.GetHost() + ":" + std::to_string(instance.GetPort());
}

std::string ResolvedEndpoint::GetResolvedEndpoint(const std::string& endpoint) {
  std::string polaris_addr = GetAddrWithPolaris(endpoint);
  if (!polaris_addr.empty()) {
    return polaris_addr;
  } else {
    return endpoint;
  }
}
}  // namespace ksana_llm
