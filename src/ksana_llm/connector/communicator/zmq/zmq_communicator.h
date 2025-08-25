/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>
#include "ksana_llm/connector/communicator/communicator.h"
#include "ksana_llm/connector/config.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "zmq_addon.hpp"

namespace ksana_llm {

/**
 * @struct ZmqDeviceResource
 * @brief Resources for ZeroMQ communication for a specific device
 */
struct ZmqDeviceResource {
  /** @brief Socket for sending messages */
  std::unique_ptr<zmq::socket_t> send_socket;

  /** @brief Rank of the sender in the communication group */
  int send_rank;

  /** @brief Rank of the receiver in the communication group */
  int recv_rank;

  /** @brief Whether the resource is active/valid */
  bool is_active = false;

  std::string ToString() const {
    std::string result = "ZmqDeviceResource send_rank: " + std::to_string(send_rank) +
                         ", recv_rank: " + std::to_string(recv_rank) + ", is_active: " + (is_active ? "true" : "false");

    // 安全地获取套接字端点信息
    if (send_socket) {
      std::string endpoint = send_socket->get(zmq::sockopt::last_endpoint);
      result += " " + endpoint;
    } else {
      result += " None";
    }
    return result;
  }
};

/**
 * @struct ZmqCommGroup
 * @brief Group of ZeroMQ communication resources
 */
struct ZmqCommGroup {
  /** @brief Resources for each device in the group */
  std::vector<std::unique_ptr<ZmqDeviceResource>> device_resources;

  /** @brief Address tuples for all nodes in the communication group */
  std::vector<std::tuple<int, int, std::string>> address_tuples;

  /** @brief Whether this communication group is active */
  bool IsActive() const { return !device_resources.empty(); }
};

/**
 * @class ZmqCommunicator
 * @brief ZeroMQ-based implementation of the Communicator
 *
 * Manages communication between nodes using ZeroMQ.
 */
class ZmqCommunicator : public Communicator {
  // Friend class for testing access to private members
  friend class TestableZmqCommunicator;

 public:
  explicit ZmqCommunicator(const ksana_llm::ConnectorConfig& config) : Communicator(config) {}
  ~ZmqCommunicator() override;

  void Shutdown() override;
  Status Initialize() override;

  // Communicator 接口实现
  Status Send(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, const void* buf,
              size_t count, DataType dtype) override;
  Status Recv(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, void* buf, size_t count,
              DataType dtype) override {
    return Status(RetCode::RET_NOT_IMPLEMENTED, "Recv not implemented in ZmqCommunicator");
  };

  Status SendGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<const void*>& buffers, const std::vector<size_t>& counts,
                   DataType dtype) override {
    return Status(RetCode::RET_NOT_IMPLEMENTED, "SendGroup not implemented in ZmqCommunicator");
  }
  Status RecvGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<void*>& buffers, const std::vector<size_t>& counts, DataType dtype) override {
    return Status(RetCode::RET_NOT_IMPLEMENTED, "RecvGroup not implemented in ZmqCommunicator");
  };

  /**
   *
   * @param comm_group_to_id Map of communication group keys to communication IDs
   * @param comm_group_to_address Map of communication group keys to address information
   * @return Status indicating success or failure
   */
  Status ProcessHeartbeatData(const std::unordered_map<std::string, std::string>& comm_group_to_id,
                              const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&
                                  comm_group_to_address) override;

  /**
   * @brief Protected implementation of SetReceiveCallback (implements Communicator interface)
   *
   * @param callback Function to call when data is received
   */
  void DoSetReceiveCallback(const ReceiveCallback& callback) override;

  /**
   * @brief Set callback for sending communication IDs
   * @param callback Function to call to send a communication ID
   * This is a no-op for ZmqCommunicator, provided for interface consistency.
   */
  void SetSendCommIdCallback(std::function<Status(const std::string&, const std::string&)> callback);

  /**
   * @brief Calculate ranks for ZMQ communication
   *
   * @param group_role Role of this node (PREFILL or DECODE)
   * @param node_rank Rank of this node
   * @param world_size Total number of nodes
   * @param device_count Number of tensor parallel devices
   * @param dev_id Device ID within the node
   * @return Tuple of (current rank, peer rank, send rank, recv rank)
   */
  static std::tuple<int, int, int, int> CalcZmqRanks(GroupRole group_role, int node_rank, int world_size,
                                                     int device_count, int dev_id);

  /**
   * @brief Check if a connection is ready for communication
   *
   * @param group_key Key identifying the communication group
   * @param dev_id Device ID within the group
   * @return true if connection is ready, false otherwise
   */
  bool IsConnectionReady(const std::string& group_key, int dev_id) const;

 private:
  /**
   * @brief Create a ZeroMQ communication group
   *
   * @param group_key Key identifying the communication group
   * @param address_tuples Address tuples for all nodes in the communication group
   * @return Status indicating success or failure
   */
  Status CreateCommGroup(const std::string& group_key,
                         const std::vector<std::tuple<int, int, std::string>>& address_tuples);

  /**
   * @brief Check if a communication group is available
   *
   * @param group_key Key identifying the communication group
   * @return true if available, false otherwise
   */
  bool CheckCommGroupAvailable(const std::string& group_key);

  /**
   * @brief Create device resources for a communication group
   *
   * @param group_key Key identifying the communication group
   * @return Status indicating success or failure
   */
  Status CreateDeviceResources(const std::string& group_key);

  /**
   * @brief ZeroMQ receive loop (runs in a separate thread)
   */
  void ReceiveLoop();

  /**
   * @brief Helper method for tests to set communication group
   *
   * @param group_key Key identifying the communication group
   * @param group Pointer to the communication group
   */
  void SetCommGroupForTest(const std::string& group_key, std::unique_ptr<ZmqCommGroup> group) {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(group);
  }

 private:
  /** @brief Flag indicating if the receive thread should continue running */
  std::atomic<bool> running_{false};

  /** @brief Thread for receiving messages */
  std::thread recv_thread_;

  /** @brief Socket for receiving messages */
  std::unique_ptr<zmq::socket_t> socket_;

  /** @brief ZeroMQ context */
  std::unique_ptr<zmq::context_t> zmq_ctx_;

  /** @brief Callback function for sending communication IDs */
  std::function<Status(const std::string&, const std::string&)> send_comm_id_callback_;

  /** @brief Communication groups managed by this communicator */
  std::unordered_map<std::string, std::unique_ptr<ZmqCommGroup>> comm_groups_;

  /** @brief Mutex for thread safety */
  std::mutex comm_group_mutex_;

  /** @brief Callback for received messages (modern C++ style) */
  ReceiveCallback receive_callback_;

  void* recv_user_data_ = nullptr;
};

}  // namespace ksana_llm
