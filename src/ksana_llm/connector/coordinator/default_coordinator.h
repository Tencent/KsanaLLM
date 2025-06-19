/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/coordinator/coordinator.h"
#include "ksana_llm/connector/coordinator/default_coordinator.h"
#include "ksana_llm/connector/device_collector.h"
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/connector/router_client/http_router_client.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
/**
 * @class DefaultCoordinator
 * @brief ICoordinator接口的默认实现，负责节点注册、心跳管理、通信组信息获取等
 */
class DefaultCoordinator : public Coordinator {
 public:
  using HeartbeatResponseCallback = Coordinator::HeartbeatResponseCallback;

  DefaultCoordinator(const ConnectorConfig& config, std::shared_ptr<RouterClient> router_client);
  ~DefaultCoordinator() override;

  Status Initialize() override;
  bool IsInitialized() const;
  void Shutdown() override;
  Status StartHeartbeat() override;
  void StopHeartbeat() override;
  void OnHeartbeatResponseCallback(HeartbeatResponseCallback cb) override;
  int GetPortFromEnvOrDefault(const std::string& env_var_name, int default_port);
  Status SendCommId(const std::string& comm_key, const std::string& comm_id);
  const KVNodeInfo& GetNodeInfo() const override;
  Status SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) override;

 private:
  ConnectorConfig config_;
  std::shared_ptr<RouterClient> router_client_;
  HeartbeatResponseCallback comm_setup_callback_;
  KVNodeInfo node_info_;
  bool heartbeat_ = false;
  std::shared_ptr<std::thread> heartbeat_thread_;

  Status RegisterNode() override;
  void HeartbeatThread() override;

  friend class TestCoordinator;
};

}  // namespace ksana_llm