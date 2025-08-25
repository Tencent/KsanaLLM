/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/communicator/zmq/zmq_communicator.h"
#include <chrono>
#include <thread>
#include "ksana_llm/utils/socket_util.h"

namespace ksana_llm {

ZmqCommunicator::~ZmqCommunicator() {
  try {
    Shutdown();
  } catch (...) {
    // Ignore all exceptions during destruction
  }
}

void ZmqCommunicator::Shutdown() {
  // Ensure idempotency - only shutdown once
  if (!running_.load()) {
    return;
  }

  KLLM_LOG_INFO << "ZmqCommunicator shutdown starting...";

  // Stop new operations
  running_.store(false);

  // Wait for receive thread to finish
  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }

  // Clear communication groups (RAII will handle socket cleanup)
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_.clear();
  }

  // Clear main socket (RAII will handle cleanup)
  socket_.reset();

  // Deliberately leak the ZMQ context to avoid zmq_ctx_term() mutex corruption
  // This is a known workaround for ZMQ shutdown issues during global destruction
  // The OS will clean up the resources when the process exits
  zmq_ctx_ = nullptr;

  KLLM_LOG_INFO << "ZmqCommunicator shutdown complete";
}

Status ZmqCommunicator::Initialize() {
  // Get port number
  int coordinator_port = config_.coordinator_port;
  if (coordinator_port < 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid coordinator port: " + std::to_string(coordinator_port));
  }
  running_ = true;
  // Start ZeroMQ socket and bind to port
  // Use raw pointer to avoid automatic context termination during shutdown
  zmq_ctx_ = std::make_unique<zmq::context_t>(1);
  socket_ = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::rep);
  std::string interface;
  std::string ip;
  Status status = GetAvailableInterfaceAndIP(interface, ip);
  if (!status.OK() || ip.empty()) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to get available IP address");
  }
  std::string bind_addr = "tcp://" + ip + ":" + std::to_string(coordinator_port);
  socket_->bind(bind_addr);
  KLLM_LOG_INFO << "ZeroMQ REP socket bound to " << bind_addr;
  recv_thread_ = std::thread(&ZmqCommunicator::ReceiveLoop, this);

  KLLM_LOG_INFO << "ZmqCommunicator initialized successfully";
  return Status();
}

Status ZmqCommunicator::ProcessHeartbeatData(
    const std::unordered_map<std::string, std::string>& comm_group_to_id,
    const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address) {
  KLLM_LOG_DEBUG << "Processing heartbeat data for ZMQ communicator";
  if (comm_group_to_address.empty()) {
    KLLM_LOG_DEBUG << "No communication groups provided in heartbeat response";
    return Status();
  }

  // 处理地址信息
  for (const auto& [group_key, address_tuples] : comm_group_to_address) {
    KLLM_LOG_DEBUG << "Processing communication group: " << group_key;
    if (!CheckCommGroupAvailable(group_key)) {
      KLLM_LOG_DEBUG << "Communication group " << group_key << " is not available";
      if (static_cast<int>(address_tuples.size()) != 2 * world_size_) {
        KLLM_LOG_INFO << "Address tuples size mismatch for group " << group_key << ": " << address_tuples.size()
                      << " vs " << (2 * world_size_) << " waiting regsiter";
        continue;
      }

      Status status = CreateCommGroup(group_key, address_tuples);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "Failed to create communication group " << group_key << ": " << status.GetMessage();
        continue;
      }
    }

    auto it = comm_groups_.find(group_key);
    if (it != comm_groups_.end()) {
      auto& comm_group = it->second;
      if (comm_group && !comm_group->IsActive()) {
        Status status = CreateDeviceResources(group_key);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "Failed to create device resources for group " << group_key << ": " << status.GetMessage();
          continue;
        }
        KLLM_LOG_INFO << "ZeroMQ communication group " << group_key << " initialized";
      } else {
        KLLM_LOG_DEBUG << "ZeroMQ communication group " << group_key << " is already active";
      }
    }
  }

  return Status();
}

std::tuple<int, int, int, int> ZmqCommunicator::CalcZmqRanks(GroupRole group_role, int node_rank, int world_size,
                                                             int device_count, int dev_id) {
  //   int global_world_size = 2 * world_size * device_count;
  int role = (group_role == GroupRole::PREFILL) ? 0 : 1;
  int peer_role = 1 - role;

  // Calculate ranks
  int cur_rank = role * world_size + node_rank * device_count + dev_id;
  int peer_rank = peer_role * world_size + node_rank * device_count + dev_id;
  int send_rank = cur_rank;
  int recv_rank = peer_rank;

  return std::make_tuple(cur_rank, peer_rank, send_rank, recv_rank);
}

Status ZmqCommunicator::CreateCommGroup(const std::string& group_key,
                                        const std::vector<std::tuple<int, int, std::string>>& address_tuples) {
  // Create a new communication group
  auto comm_group = std::make_unique<ZmqCommGroup>();
  comm_group->address_tuples = address_tuples;

  // Store the new communication group
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(comm_group);
  }

  return Status();
}

bool ZmqCommunicator::CheckCommGroupAvailable(const std::string& group_key) {
  std::lock_guard<std::mutex> lock(comm_group_mutex_);
  auto it = comm_groups_.find(group_key);
  return it != comm_groups_.end() && it->second && it->second->IsActive();
}

bool ZmqCommunicator::IsConnectionReady(const std::string& group_key, int dev_id) const {
  // 使用 const_cast 来解决常量方法中使用非常量互斥锁的问题
  std::lock_guard<std::mutex> lock(*const_cast<std::mutex*>(&comm_group_mutex_));
  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second) {
    return false;
  }

  if (dev_id < 0 || static_cast<size_t>(dev_id) >= it->second->device_resources.size()) {
    return false;
  }

  auto& device_resource = it->second->device_resources[dev_id];
  return device_resource && device_resource->send_socket && device_resource->is_active;
}

Status ZmqCommunicator::CreateDeviceResources(const std::string& group_key) {
  // 参数合法性校验，防止负数下标
  if (node_rank_ < 0 || world_size_ <= 0 || device_count_ <= 0) {
    KLLM_LOG_ERROR << "Invalid communicator config: node_rank_=" << node_rank_ << ", world_size_=" << world_size_
                   << ", device_count_=" << device_count_;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid communicator config: negative or zero rank/size/count");
  }
  std::lock_guard<std::mutex> lock(comm_group_mutex_);
  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& comm_group = it->second;
  const auto& address_tuples = comm_group->address_tuples;

  // Calculate global world size
  const int global_world_size = address_tuples.size();
  if (global_world_size != 2 * world_size_) {
    KLLM_LOG_ERROR << "Address tuples size mismatch: " << global_world_size << " vs " << (2 * world_size_);
    return Status(RetCode::RET_INVALID_ARGUMENT, "Address tuples size mismatch");
  }

  // Create device resources
  comm_group->device_resources.resize(device_count_);
  for (int i = 0; i < device_count_; ++i) {
    // Calculate ranks for this device
    auto [cur_rank, peer_rank, send_rank, recv_rank] =
        CalcZmqRanks(config_.group_role, node_rank_, world_size_, device_count_, i);

    // Boundary check
    if (cur_rank >= global_world_size || peer_rank >= global_world_size) {
      KLLM_LOG_ERROR << "Rank out of range: cur_rank=" << cur_rank << ", peer_rank=" << peer_rank
                     << ", address_tuples.size()=" << address_tuples.size();
      return Status(RetCode::RET_INVALID_ARGUMENT, "Rank out of range");
    }

    std::string local_addr = std::get<2>(address_tuples[cur_rank]);
    std::string peer_addr = std::get<2>(address_tuples[peer_rank]);

    // Create send socket
    try {
      auto device_resource = std::make_unique<ZmqDeviceResource>();
      if (zmq_ctx_ == nullptr) {
        KLLM_LOG_ERROR << "ZMQ context is null";
        return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "ZMQ context is null");
      }
      device_resource->send_socket = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::req);
      device_resource->send_socket->connect("tcp://" + peer_addr);
      device_resource->send_rank = send_rank;
      device_resource->recv_rank = recv_rank;
      device_resource->is_active = true;

      comm_group->device_resources[i] = std::move(device_resource);
    } catch (const zmq::error_t& e) {
      KLLM_LOG_ERROR << "Failed to create ZeroMQ socket: " << e.what();
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, e.what());
    }
  }

  return Status();
}

// Communicator 接口实现 - 发送数据
Status ZmqCommunicator::Send(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                             const void* buf, size_t count, DataType dtype) {
  if (!buf || count == 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid buffer or count for Send");
  }
  int dev_id = src_dev_id;  // ZMQ通信只需要local device的相关信息
  // 查找通信组和设备资源
  std::lock_guard<std::mutex> lock(comm_group_mutex_);
  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second || dev_id < 0 ||
      static_cast<size_t>(dev_id) >= it->second->device_resources.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid group_key or dev_id for Send");
  }
  auto& device_resource = it->second->device_resources[dev_id];
  if (!device_resource || !device_resource->send_socket) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Send socket not initialized");
  }

  // 发送数据
  try {
    zmq::message_t msg(buf, count);
    KLLM_LOG_DEBUG << "[ZMQ Send] Sending " << count << " bytes to device " << dev_id << " in group " << group_key
                   << " and resource " << device_resource->ToString();

    auto send_result = device_resource->send_socket->send(msg, zmq::send_flags::none);
    if (!send_result) {
      KLLM_LOG_ERROR << "[ZMQ] Failed to send message to group_key=" << group_key << ", dev_id=" << dev_id;
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to send ZMQ message");
    }
    zmq::message_t reply;
    auto recv_result = device_resource->send_socket->recv(reply, zmq::recv_flags::none);
    (void)recv_result;  // 显式忽略返回值，消除 warning
    KLLM_LOG_DEBUG << "[ZMQ] Successfully sent " << count << " bytes to group_key=" << group_key
                   << ", dev_id=" << dev_id << ", send_rank=" << device_resource->send_rank << " and resource "
                   << device_resource->ToString();
  } catch (const zmq::error_t& e) {
    KLLM_LOG_ERROR << "[ZMQ] Error during send: " << e.what();
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, e.what());
  }

  return Status();
}

void ZmqCommunicator::DoSetReceiveCallback(const ReceiveCallback& callback) {
  // 存储回调函数以在ReceiveLoop中使用
  receive_callback_ = callback;
  KLLM_LOG_DEBUG << "Receive callback set for ZmqCommunicator";
}

void ZmqCommunicator::ReceiveLoop() {
  KLLM_LOG_DEBUG << "[ReceiveLoop] Starting receive loop...";

  while (running_.load()) {
    if (!socket_) {
      KLLM_LOG_ERROR << "[ReceiveLoop] ZMQ socket is not initialized, exiting receive loop.";
      break;
    }

    try {
      // 为了确保 socket 在 poll 期间不会被释放，我们使用较短的超时时间
      zmq::pollitem_t items[] = {{*socket_, 0, ZMQ_POLLIN, 0}};

      KLLM_LOG_DEBUG << "[ZMQ ReceiveLoop] Waiting for messages on socket...";
      int rc = zmq::poll(items, 1, std::chrono::milliseconds(50));  // 减少超时时间为 50ms

      if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
        zmq::message_t msg;
        KLLM_LOG_DEBUG << "[ZMQ ReceiveLoop] Message received, preparing to receive data...";
        auto recv_result = socket_->recv(msg, zmq::recv_flags::dontwait);  // 使用非阻塞接收
        if (recv_result && *recv_result > 0) {
          socket_->send(zmq::message_t(), zmq::send_flags::dontwait);
          std::string data(static_cast<const char*>(msg.data()), msg.size());
          // 在调用回调之前再次检查状态
          if (receive_callback_) {
            KLLM_LOG_DEBUG << "[ZMQ ReceiveLoop] Invoking receive_callback with data size=" << data.size();
            receive_callback_(data.data(), data.size(), 0, recv_user_data_);
            KLLM_LOG_DEBUG << "[ZMQ ReceiveLoop] Receive callback completed";
          } else {
            KLLM_LOG_INFO << "[ZMQ ReceiveLoop] Received data but no callback registered: " << data;
          }
        } else {
          KLLM_LOG_DEBUG << "[ReceiveLoop] No message available or failed to receive.";
        }
      } else if (rc == 0) {
        KLLM_LOG_DEBUG << "[ReceiveLoop] Poll timeout, continuing...";
      } else {
        KLLM_LOG_DEBUG << "[ReceiveLoop] Poll returned: " << rc;
      }
    } catch (const zmq::error_t& e) {
      KLLM_LOG_ERROR << "[ReceiveLoop] ZMQ error: " << e.what();
      // 对于 ZMQ 错误，我们也应该退出循环
      break;
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "[ReceiveLoop] Exception: " << e.what();
      break;
    }
  }
  KLLM_LOG_DEBUG << "[ReceiveLoop] Exiting receive loop.";
}

}  // namespace ksana_llm
