/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include <nlohmann/json.hpp>

namespace ksana_llm {
struct KVHeartbeatResponse {
  /** @brief Unique identifier for this node */
  std::string node_id;

  /** @brief Name of this node */
  std::string node_name;

  /** @brief Flag indicating if the node is online */
  bool is_online;

  /** @brief Flag indicating if the group is ready for communication */
  bool group_ready;

  /** @brief Port for coordinator service */
  int coordinator_port = 13579;

  /** @brief Role of this node (prefill or decode) */
  std::string node_role;

  /** @brief Rank of this node within its group */
  int node_rank = 0;

  /** @brief Timestamp of the heartbeat response */
  std::string timestamp;

  /**
   * @brief Map of communication addresses for each communication group
   *
   * Maps "prefill_group__decode_group" -> node_rank, device_id, ip:port
   */
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> comm_group_to_address;

  /**
   * @brief Map of communication group IDs
   *
   * Maps "prefill_group__decode_group" -> "comm_id"
   */
  std::unordered_map<std::string, std::string> comm_group_to_id;
};

/**
 * @class RouterClient
 * @brief RouterClient class for managing node registration and communication with simple_router service
 *
 * This class handles node registration, heartbeat management, and retrieval of communication groups
 * from the router service.
 */
class RouterClient {
 public:
  /**
   * @brief Construct a new RouterClient object
   * @param endpoint The URL endpoint of the router service
   */
  explicit RouterClient(const std::string& endpoint) {}

  /**
   * @brief Register node information with the router service
   * @param node_info The node information to register
   * @return Status Registration status
   */
  virtual Status RegisterNode(const KVNodeInfo& node_info) = 0;

  virtual const KVNodeInfo& GetNodeInfo() const = 0;

  virtual Status SendCommId(const std::string& node_id, const std::string& comm_key, const std::string& comm_id) = 0;
  /**
   * @brief Send heartbeat to the router service and get communication groups
   * @param response Reference to store the heartbeat response
   * @return Status Heartbeat status
   *
   * Sends a heartbeat to the router service and updates the response with
   * current communication group information.
   */
  virtual Status SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) = 0;

  /**
   * @brief Make an HTTP request (for HTTPRouterClient and testing mocks)
   * @param path The API path
   * @param method HTTP method (GET, POST, etc.)
   * @param json_data The JSON data to send
   * @return std::string The response from the service
   *
   * Default implementation throws or returns empty string; override in HTTPRouterClient/mocks.
   */
  virtual std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                      const nlohmann::json& json_data) = 0;
};
}  // namespace ksana_llm
