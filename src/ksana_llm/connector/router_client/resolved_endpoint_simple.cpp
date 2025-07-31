/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "resolved_endpoint.h"

namespace ksana_llm {

std::string ResolvedEndpoint::GetAddrWithPolaris(const std::string& endpoint) {
  // Simple implementation: not supported without internal libraries
  return "";
}

std::string ResolvedEndpoint::GetResolvedEndpoint(const std::string& endpoint) {
  // Simple implementation: just return the original endpoint
  return endpoint;
}

}  // namespace ksana_llm
