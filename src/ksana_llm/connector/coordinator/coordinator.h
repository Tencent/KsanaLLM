/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/connector/router_client/router_client.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
/**
 * @class ICoordinator
 * @brief 协调器接口，负责节点注册、心跳管理、通信组信息获取等
 */
class Coordinator {
 public:
  using HeartbeatResponseCallback =
      std::function<Status(const std::unordered_map<std::string, std::string>&,
                           const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&)>;

  Coordinator() = default;
  virtual ~Coordinator() = default;

  virtual Status Initialize() = 0;

  // 关闭协调器（停止心跳、清理资源）
  virtual void Shutdown() = 0;

  // 启动心跳线程
  virtual Status StartHeartbeat() = 0;

  // 停止心跳线程
  virtual void StopHeartbeat() = 0;

  virtual bool IsInitialized() const = 0;

  virtual Status SendCommId(const std::string& comm_key, const std::string& comm_id) = 0;

  virtual Status RegisterNode() = 0;
  virtual Status SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) = 0;
  virtual const KVNodeInfo& GetNodeInfo() const = 0;
  virtual void HeartbeatThread() = 0;
  // 设置心跳响应回调
  virtual void OnHeartbeatResponseCallback(HeartbeatResponseCallback cb) = 0;

 protected:
  bool initialized_ = false;
};

}  // namespace ksana_llm