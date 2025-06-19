/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stdint.h>

#include <string>

namespace ksana_llm {

struct NodeInfo {
  std::string host;
  uint16_t port;
};

struct NodeInfoHash {
  std::size_t operator()(const NodeInfo& node_info) const {
    auto h1 = std::hash<std::string>{}(node_info.host);
    auto h2 = std::hash<uint16_t>{}(node_info.port);
    return h1 ^ h2;
  }
};

struct NodeInfoEqual {
  bool operator()(const NodeInfo& lhs, const NodeInfo& rhs) const {
    return lhs.host == rhs.host && lhs.port == rhs.port;
  }
};

}  // namespace ksana_llm
