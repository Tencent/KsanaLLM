/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <map>
#include <string>
#include <vector>

namespace ksana_llm {

/**
 * @struct DeviceInfo
 * @brief Unified structure for storing compute device information
 *
 * This structure consolidates device information from both router_client and collect_node_info
 */
struct DeviceInfo {
  /** @brief Numerical ID of the device (e.g., GPU index) */
  int device_id;

  /** @brief Device type (e.g., "NVIDIA A100", "Ascend NPU") */
  std::string device_type;

  /** @brief IP address specifically for this device */
  std::string device_ip;

  /** @brief Default constructor */
  DeviceInfo() : device_id(0) {}

  /** @brief Constructor with essential fields */
  DeviceInfo(int id, const std::string& type, const std::string& ip)
      : device_id(id), device_type(type), device_ip(ip) {}
};

/**
 * @struct KVNodeInfo
 * @brief Unified structure for storing node information
 *
 * This structure contains all necessary information about a compute node,
 * including its identification, addressing, and available devices.
 */
struct KVNodeInfo {
  /** @brief Unique identifier for this node */
  std::string node_id;

  /** @brief Host address with port for inference (e.g. "127.0.0.1:8080") */
  std::string inference_addr = "127.0.0.1:8080";

  /** @brief Port for coordinator service */
  int coordinator_port = 13579;

  /** @brief Group name that this node belongs to */
  std::string group_name;

  /** @brief Role of this node (prefill or decode) */
  std::string group_role;

  /** @brief Rank of this node within its group */
  int node_rank = 0;

  /** @brief Total number of nodes in the group */
  int world_size = 1;

  /** @brief Flag indicating if the node is online */
  bool is_online = false;

  /** @brief Timestamp of the last heartbeat */
  std::string last_heartbeat;

  /** @brief Communication ID for this node */
  std::string comm_id;

  /** @brief List of devices available on this node */
  std::vector<DeviceInfo> devices;

  /** @brief Start time of the node */
  std::string start_time;

  /** @brief Job ID for the deployment */
  std::string job_id;
};

}  // namespace ksana_llm
