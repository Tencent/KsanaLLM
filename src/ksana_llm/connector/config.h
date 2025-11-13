/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace ksana_llm {
/**
 * @enum CommunicationType
 * @brief Defines the available communication protocols for the connector
 */
enum class CommunicationType {
  NCCL, /**< NVIDIA Collective Communications Library for GPU communication */
  TCP,  /**< Transmission Control Protocol for general network communication */
  MPI,  /**< Message Passing Interface for distributed computing */
  SHM,  /**< Shared Memory for inter-process communication on the same machine */
  ZMQ   /**< Remote Procedure Call for service-oriented communication */
};

/**
 * @enum GroupRole
 * @brief Defines the role of a node in the prefill-decode pipeline
 */
enum class GroupRole {
  PREFILL, /**< Node responsible for prefill operations */
  DECODE,  /**< Node responsible for decode operations */
  BOTH,    /**< Node performs both prefill and decode operations */
  NONE     /**< No specific role assigned */
};

/**
 * @brief Convert GroupRole enum to string representation
 * @param role The GroupRole enum value to convert
 * @return String representation of the role
 */
inline std::string GroupRoleToString(GroupRole role) {
  switch (role) {
    case GroupRole::NONE:
      return "none";
    case GroupRole::PREFILL:
      return "prefill";
    case GroupRole::DECODE:
      return "decode";
    default:
      return "none";
  }
}
/**
 * @struct ConnectorConfig
 * @brief Configuration parameters for connector initialization
 *
 * Contains all the necessary parameters to set up a communication connector
 * between prefill and decode components in the KsanaLLM framework.
 */
struct ConnectorConfig {
  /** @brief Router address (e.g., "http://localhost:8000") */
  std::string router_addr;

  /**
   * @brief Cluster name for this connector
   *
   * All prefill and decode nodes under the same cluster name will be grouped together
   * and every prefill node will communicate with decode nodes in the same cluster.
   */
  std::string cluster_name;

  /** @brief Connector role (prefill, decode, or both) */
  GroupRole group_role = GroupRole::NONE;

  /** @brief Type of communication protocol to use */
  CommunicationType communication_type = CommunicationType::NCCL;

  /** @brief Heartbeat interval in milliseconds */
  int heartbeat_interval_ms = 5000;

  std::string coordinator_addr = ""; /**< @brief Address for coordinator service */

  std::string inference_addr = ""; /**< @brief Inference address (IP:port format) */

  int node_rank = -1;
  int device_count = -1;           /**< @brief Number of devices in the group, used for attention. */
  int world_size = -1;             /**< @brief Total number of devices for group prefill and decode. */
  int transfer_batch = 1048576;    /**< @brief Batch size for transfer */
  int send_thread_num = 4;         /**< @brief Number of threads for sending tasks */
  int connector_waiting_sec = 3;   /**< @brief Timeout for connector operations in milliseconds */
  int circular_bucket_size = 8192; /**< @brief Size of the circular buffer for task keys */
  int circular_bucket_num = 4;     /**< @brief Number of buckets for task management, used for circular buffer */
  int circular_thread_num = 4;     /**< @brief Number of threads for TBB task arena, used for parallel operations */

  /**
   * @brief Convert CommunicationType enum to string representation
   * @param type The CommunicationType enum value to convert
   * @return String representation of the communication type
   */
  static std::string CommunicationTypeToString(CommunicationType type) {
    switch (type) {
      case CommunicationType::NCCL:
        return "NCCL";
      case CommunicationType::ZMQ:
        return "ZMQ";
      default:
        return "unknown";
    }
  }

  /**
   * @brief Convert the configuration to a string representation for debugging
   * @return String representation of the configuration
   */
  std::string toString() const {
    std::string result = "ConnectorConfig {\n";
    result += "  router_addr: " + router_addr + "\n";
    result += "  cluster_name: " + cluster_name + "\n";
    result += "  communication_type: " + CommunicationTypeToString(communication_type) + "\n";
    result += "  heartbeat_interval_ms: " + std::to_string(heartbeat_interval_ms) + "\n";
    result += "  coordinator_addr: " + coordinator_addr + "\n";
    result += "  inference_addr: " + inference_addr + "\n";
    result += "  node_rank: " + std::to_string(node_rank) + "\n";
    result += "  device_count: " + std::to_string(device_count) + "\n";
    result += "  group_role: " + GroupRoleToString(group_role) + "\n";
    result += "  connector_waiting_sec: " + std::to_string(connector_waiting_sec) + "\n";
    result += "  send_thread_num: " + std::to_string(send_thread_num) + "\n";
    result += "  world_size: " + std::to_string(world_size) + "\n";
    result += "  transfer_batch: " + std::to_string(transfer_batch) + "\n";
    result += "}";
    return result;
  }
};

}  // namespace ksana_llm