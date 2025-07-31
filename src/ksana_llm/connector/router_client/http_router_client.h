/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/connector/router_client/router_client.h"
#include "ksana_llm/connector/router_client/resolved_endpoint.h"

namespace ksana_llm {

/**
 * @class HTTPRouterClient
 * @brief 基于HTTP的路由器客户端实现
 */
class HTTPRouterClient : public RouterClient {
 public:
  /**
   * @brief Construct a new RouterClient object
   * @param endpoint The URL endpoint of the router service
   */
  explicit HTTPRouterClient(const std::string& endpoint) : RouterClient(endpoint), endpoint_(endpoint) {
  }

  /**
   * @brief Check if the router service is reachable and responding
   * @return Status Success if the service is responding, error otherwise
   */
  Status CheckConnectivity();

  /**
   * @brief Register node information with the router service
   * @param node_info The node information to register
   * @return Status Registration status
   */
  Status RegisterNode(const KVNodeInfo& node_info) override;

  /**
   * @brief Send heartbeat to the router service and get communication groups
   * @param response Reference to store the heartbeat response
   * @return Status Heartbeat status
   *
   * Sends a heartbeat to the router service and updates the response with
   * current communication group information.
   */
  Status SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) override;

  /**
   * @brief Register NCCL ID for a specific communication group
   * @param node_id The ID of the node registering the NCCL ID
   * @param comm_key The communication group key (format: "prefill_group_decode_group")
   * @param comm_id The NCCL ID to register
   * @return Status Registration status and returned NCCL ID
   *
   * Registers a communication ID for a specific group with the router service.
   */
  Status SendCommId(const std::string& node_id, const std::string& comm_key, const std::string& comm_id) override;

  /**
   * @brief Helper method to make HTTP request to the router service
   * @param path The API path
   * @param method HTTP method (GET, POST, etc.)
   * @param json_data The JSON data to send
   * @return std::string The response from the service
   *
   * Makes an HTTP request to the router service with the specified path, method, and data.
   */
  std::string MakeHttpRequest(const std::string& path, const std::string& method,
                              const nlohmann::json& json_data) override;

  /**
   * @brief Get the registered node info
   * @return const KVNodeInfo& The registered node information
   */
  const KVNodeInfo& GetNodeInfo() const override { return node_info_; }

  // 生成唯一任务ID（内部工具函数）
  std::string GenerateTaskID();


 private:
  /** @brief Endpoint URL for the router service */
  std::string endpoint_;

  /** @brief Store the registered node info */
  KVNodeInfo node_info_;
};

}  // namespace ksana_llm
