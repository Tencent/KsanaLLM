/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace ksana_llm {
class ResolvedEndpoint {
 public:
  ResolvedEndpoint() = default;
  static std::string GetAddrWithPolaris(const std::string& endpoint);
  static std::string GetResolvedEndpoint(const std::string& endpoint);
};

}  // namespace ksana_llm
