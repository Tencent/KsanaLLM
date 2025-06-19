/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "base64.hpp"
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/coordinator/coordinator.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

/**
 * @class Communicator
 * @brief Pure abstract base class for all communicator implementations
 *
 * This class defines the interface that all communicator implementations must adhere to.
 * It uses modern C++ features like std::function for callbacks and provides a clean,
 * consistent interface for communication operations.
 */
class Communicator {
 public:
  /**
   * @brief Constructor
   * @param config Configuration for communication
   * @param node_rank Rank of this node
   * @param world_size Total number of nodes
   * @param device_count Number of devices used for tensor parallelism
   */
  explicit Communicator(const ksana_llm::ConnectorConfig& config)
      : config_(config),
        node_rank_(config.node_rank),
        world_size_(config.world_size),
        device_count_(config.device_count) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~Communicator() = default;

  /**
   * @brief Shutdown the communicator
   */
  virtual void Shutdown() = 0;

  /**
   * @brief Initialize the communicator
   * @return Status indicating success or failure
   */
  virtual Status Initialize() = 0;

  /**
   * @brief Send data to a peer
   *
   * @param group_key Key identifying the communication group
   * @param dev_id Device ID within the node
   * @param buf Pointer to data to be sent
   * @param count Number of elements to send
   * @param dtype Data type of elements (may be nullptr)
   * @param stream CUDA stream to use (may be nullptr)
   * @return Status indicating success or failure
   */
  virtual Status Send(const std::string& group_key, int dev_id, uint64_t job_id, const void* buf, size_t count,
                      DataType dtype) = 0;

  /**
   * @brief Receive data from a peer
   *
   * @param group_key Key identifying the communication group
   * @param dev_id Device ID within the node
   * @param buf Pointer to buffer for received data
   * @param count Number of elements to receive
   * @param dtype Data type of elements (may be nullptr)
   * @param stream CUDA stream to use (may be nullptr)
   * @return Status indicating success or failure
   */
  virtual Status Recv(const std::string& group_key, int dev_id, uint64_t job_id, void* buf, size_t count,
                      DataType dtype) = 0;

  /**
   * @brief Send a group of data items in a single operation
   *
   * @param group_key Key identifying the communication group
   * @param dev_id Device ID within the node
   * @param buffers Vector of data buffers to send
   * @param counts Vector of element counts for each buffer
   * @param dtype Data type of elements (may be nullptr)
   * @param stream CUDA stream to use (may be nullptr)
   * @return Status indicating success or failure
   */
  virtual Status SendGroup(const std::string& group_key, int dev_id, uint64_t job_id,
                           const std::vector<const void*>& buffers, const std::vector<size_t>& counts,
                           DataType dtype) = 0;

  /**
   * @brief Receive a group of data items in a single operation
   *
   * @param group_key Key identifying the communication group
   * @param dev_id Device ID within the node
   * @param buffers Vector of data buffers to receive into
   * @param counts Vector of element counts for each buffer
   * @param dtype Data type of elements (may be nullptr)
   * @param stream CUDA stream to use (may be nullptr)
   * @return Status indicating success or failure
   */
  virtual Status RecvGroup(const std::string& group_key, int dev_id, uint64_t job_id, const std::vector<void*>& buffers,
                           const std::vector<size_t>& counts, DataType dtype) = 0;

  virtual bool IsConnectionReady(const std::string& group_key, int dev_id) const = 0;
  /**
   * @brief Process heartbeat data forwarded from CommunicatorManager
   *
   * @param comm_group_to_id Map of group keys to communication IDs
   * @param comm_group_to_address Map of group keys to address tuples
   * @return Status indicating success or failure
   */
  virtual Status ProcessHeartbeatData(
      const std::unordered_map<std::string, std::string>& comm_group_to_id,
      const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address) = 0;

  /**
   * @brief Type definition for receive callback function
   */
  using ReceiveCallback = std::function<void(const char* data, size_t len, uint64_t job_id, void* user_data)>;

  /**
   * @brief Set callback for received data (template version for perfect forwarding)
   *
   * @tparam Callback Type of the callback
   * @param callback Function to call when data is received
   */
  template <typename Callback>
  void SetReceiveCallback(Callback&& callback) {
    // Convert to std::function and delegate to the non-template version
    DoSetReceiveCallback(ReceiveCallback(std::forward<Callback>(callback)));
  }

 protected:
  /**
   * @brief Protected implementation of SetReceiveCallback
   *
   * @param callback Function to call when data is received
   */
  virtual void DoSetReceiveCallback(const ReceiveCallback& callback) = 0;

 protected:
  /** @brief Configuration for this communicator */
  ConnectorConfig config_;

  /** @brief Rank of this node in the distributed system */
  int node_rank_;

  /** @brief Total number of nodes in the distributed system */
  int world_size_;

  /** @brief Number of devices used for tensor parallelism */
  int device_count_;
};

}  // namespace ksana_llm