/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#include <numeric>
#include "base64.hpp"

namespace ksana_llm {
#ifdef ENABLE_CUDA
NcclCommunicator::~NcclCommunicator() { Shutdown(); }

void NcclCommunicator::DoSetReceiveCallback(const ReceiveCallback& callback) { receive_callback_ = callback; }

void NcclCommunicator::Shutdown() {
  // Close all NCCL resources
  for (auto& [group_key, comm_group] : comm_groups_) {
    KLLM_LOG_INFO << "Shutting down NCCL group: " << group_key;
    // The NcclCommGroup destructor will handle resource cleanup
  }
  comm_groups_.clear();

  KLLM_LOG_INFO << "NcclCommunicator shutdown complete";
}

Status NcclCommunicator::Initialize() {
  // NCCL通信器的初始化
  KLLM_LOG_INFO << "NcclCommunicator initialized successfully";
  return Status();
}

NcclCommunicator::NcclRankInfo NcclCommunicator::CalcNcclRanks(GroupRole group_role, int node_rank, int world_size,
                                                               int device_count, int dev_id) {
  NcclRankInfo info{};
  // int global_world_size = 2 * world_size * device_count; // unused, remove warning
  int role = (group_role == GroupRole::PREFILL) ? 0 : 1;
  int peer_role = 1 - role;
  info.cur_rank = role * world_size + node_rank * device_count + dev_id;
  info.peer_rank = peer_role * world_size + node_rank * device_count + dev_id;
  info.send_rank = info.peer_rank;
  info.recv_rank = info.peer_rank;
  KLLM_LOG_DEBUG << "NCCL ranks: " << "cur_rank=" << info.cur_rank << ", peer_rank=" << info.peer_rank
                 << ", send_rank=" << info.send_rank << ", recv_rank=" << info.recv_rank;
  return info;
}

// NCCL设备资源工厂，仅创建NCCL相关资源，确保异常安全
std::unique_ptr<NcclDeviceResource> NcclCommunicator::CreateNcclDeviceResource(const ConnectorConfig& config,
                                                                               const ncclUniqueId& comm_id,
                                                                               int node_rank, int world_size,
                                                                               int device_count, int dev_id) {
  auto resource = std::make_unique<NcclDeviceResource>();
  // 计算全局 world size（两组）
  const int global_world_size = 2 * world_size;
  NcclRankInfo ranks = CalcNcclRanks(config.group_role, node_rank, world_size, device_count, dev_id);

  // NCCL communicator 初始化 - 使用单个通信器进行发送和接收
  cudaStream_t send_stream = nullptr;
  cudaStream_t recv_stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&send_stream));
  CUDA_CHECK(cudaStreamCreate(&recv_stream));

  ncclComm_t nccl_comm = nullptr;

  // 使用当前节点的rank初始化单个NCCL通信器
  CUDA_CHECK(cudaSetDevice(dev_id));
  ncclResult_t result = ncclCommInitRank(&nccl_comm, global_world_size, comm_id, ranks.cur_rank);
  if (result != ncclSuccess) {
    KLLM_LOG_ERROR << "Failed to initialize NCCL communicator: " << ncclGetErrorString(result);
  }

  // 对于发送和接收都使用同一个通信器
  resource->send_comm = nccl_comm;
  resource->recv_comm = nccl_comm;
  resource->send_stream = send_stream;
  resource->recv_stream = recv_stream;
  resource->send_rank = ranks.send_rank;
  resource->recv_rank = ranks.recv_rank;

  return resource;
}

bool NcclCommunicator::CheckCommGroupAvailable(const std::string& group_key) {
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    auto existing_it = comm_groups_.find(group_key);
    if (existing_it != comm_groups_.end() && existing_it->second) {
      if (existing_it->second->IsActive()) {
        KLLM_LOG_DEBUG << "Communication group " << group_key << " is already active";
        return true;
      } else {
        comm_groups_.erase(group_key);
        return false;
      }
    }
    return false;
  }
}

void NcclCommunicator::CreateNcclCommGroup(const std::string& group_key, const std::string& encoded_value) {
  // Decode the NCCL ID from base64
  std::string decoded_value = base64::from_base64(encoded_value);
  if (decoded_value.size() != sizeof(ncclUniqueId)) {
    KLLM_LOG_ERROR << "Invalid NCCL ID size for group " << group_key;
    return;
  }
  ncclUniqueId comm_id;
  memset(&comm_id, 0, sizeof(ncclUniqueId));
  memcpy(&comm_id, decoded_value.data(), sizeof(ncclUniqueId));

  // Create a new communication group
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = comm_id;
  comm_group->comm_id_str = encoded_value;

  // Store the new communication group
  {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(comm_group);
  }
}

Status NcclCommunicator::CreateDeviceResources(const std::string& group_key) {
  auto comm_group = GetCommunicatorGroup(group_key);
  if (comm_group == nullptr) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  comm_group->device_resources_.resize(device_count_);
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < device_count_; ++i) {
    try {
      cudaSetDevice(i);
      comm_group->device_resources_[i] =
          CreateNcclDeviceResource(config_, comm_group->comm_id, node_rank_, world_size_, device_count_, i);
    } catch (const std::exception& e) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                    std::string("Failed to create NCCL device resource: ") + e.what());
    }
  }
  NCCL_CHECK(ncclGroupEnd());

  return Status();
}

Status NcclCommunicator::ProcessHeartbeatData(
    const std::unordered_map<std::string, std::string>& comm_group_to_id,
    const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address) {
  if (comm_group_to_id.empty()) {
    KLLM_LOG_DEBUG << "No communication groups provided in heartbeat response";
    return Status();
  }

  // 处理每个通信组
  for (const auto& [group_key, encoded_value] : comm_group_to_id) {
    // 如果通信ID为空且是prefill角色的主节点，则需要注册一个新的ID
    std::string actual_encoded_value = encoded_value;
    if (encoded_value.empty()) {
      if (config_.group_role == GroupRole::PREFILL && node_rank_ == 0) {
        // 生成新的NCCL ID
        ncclUniqueId nccl_id;
        memset(&nccl_id, 0, sizeof(ncclUniqueId));
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        std::string to_id_bytes(reinterpret_cast<const char*>(&nccl_id), sizeof(ncclUniqueId));
        actual_encoded_value = base64::to_base64(to_id_bytes);

        // 通过回调发送新的ID
        if (send_comm_id_callback_) {
          auto status = send_comm_id_callback_(group_key, actual_encoded_value);
          if (!status.OK()) {
            KLLM_LOG_ERROR << "Failed to send NCCL ID through callback: " << status.GetMessage();
            continue;
          }
        } else {
          KLLM_LOG_ERROR << "No send_comm_id_callback_ set, cannot register NCCL ID for group: " << group_key;
          continue;
        }
      } else {
        // 非主节点或非prefill角色不处理空ID组
        continue;
      }
    }

    // 检查并创建通信组
    if (!CheckCommGroupAvailable(group_key)) {
      KLLM_LOG_DEBUG << "Communication group " << group_key << " is not available";
      CreateNcclCommGroup(group_key, actual_encoded_value);
    }

    // 如果通信组存在但尚未激活，则创建设备资源
    auto it = comm_groups_.find(group_key);
    if (it != comm_groups_.end()) {
      auto& comm_group = it->second;
      if (comm_group && !comm_group->IsActive()) {
        Status status = CreateDeviceResources(group_key);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "Failed to create device resources for group " << group_key << ": " << status.GetMessage();
          continue;
        }
        KLLM_LOG_INFO << "NCCL communication group " << group_key << " initialized";
      }
    }
  }

  return Status();
}

bool NcclCommunicator::IsConnectionReady(const std::string& group_key, int dev_id) const {
  // 使用 const_cast 来解决常量方法中使用非常量互斥锁的问题
  std::lock_guard<std::mutex> lock(*const_cast<std::mutex*>(&comm_group_mutex_));
  auto it = comm_groups_.find(group_key);
  if (it == comm_groups_.end() || !it->second) {
    return false;
  }
  return it->second->IsDeviceResourceValid(dev_id);
}

// Communicator 接口实现 - 发送数据
Status NcclCommunicator::Send(const std::string& group_key, int dev_id, uint64_t job_id, const void* buf, size_t count,
                              DataType dtype) {
  return Send(group_key, dev_id, dev_id, nullptr, buf, count, dtype);
}

// Communicator 接口实现 - 接收数据
Status NcclCommunicator::Recv(const std::string& group_key, int dev_id, uint64_t job_id, void* buf, size_t count,
                              DataType dtype) {
  return Recv(group_key, dev_id, dev_id, nullptr, buf, count, dtype);
}

// Communicator 接口实现 - 批量发送数据
Status NcclCommunicator::SendGroup(const std::string& group_key, int dev_id, uint64_t job_id,
                                   const std::vector<const void*>& buffers, const std::vector<size_t>& counts,
                                   DataType dtype) {
  return SendGroup(group_key, dev_id, dev_id, nullptr, buffers, counts, dtype);
}

// Communicator 接口实现 - 批量接收数据
Status NcclCommunicator::RecvGroup(const std::string& group_key, int dev_id, uint64_t job_id,
                                   const std::vector<void*>& buffers, const std::vector<size_t>& counts,
                                   DataType dtype) {
  return RecvGroup(group_key, dev_id, dev_id, nullptr, buffers, counts, dtype);
}

Status NcclCommunicator::Send(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                              const void* buf, size_t count, DataType dtype) {
  KLLM_LOG_DEBUG << "[NCCL] Sending data to group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", count=" << count << ", from node_rank=" << node_rank_;

  auto comm_group = NcclCommunicator::GetCommunicatorGroup(group_key);
  if (comm_group == nullptr || local_dev_id < 0 || local_dev_id >= device_count_ || peer_dev_id < 0 ||
      peer_dev_id >= device_count_) {
    KLLM_LOG_ERROR << "[NCCL] Communication group not found: " << group_key;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& resource = comm_group->device_resources_[local_dev_id];
  if (!resource || !resource->send_comm) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Send communicator not initialized");
  }

  cudaStream_t cur_stream;
  if (stream == nullptr) {
    // 如果没有提供流，则使用默认的设备流
    cur_stream = resource->send_stream;
  } else {
    cur_stream = stream;
  }
  int recv_rank;
  if (local_dev_id == peer_dev_id) {
    recv_rank = resource->recv_rank;
  } else {
    recv_rank = comm_group->device_resources_[peer_dev_id]->recv_rank;
  }

  // 执行NCCL发送操作 - 使用peer_rank作为目标
  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  NCCL_CHECK(ncclSend(buf, count, nccl_dtype, recv_rank, resource->send_comm, cur_stream));
  if (!stream) CUDA_CHECK(cudaStreamSynchronize(cur_stream));

  return Status();
}

Status NcclCommunicator::Recv(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                              void* buf, size_t count, DataType dtype) {
  KLLM_LOG_DEBUG << "[NCCL] Receiving data from group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", count=" << count << ", to node_rank=" << node_rank_;
  auto comm_group = NcclCommunicator::GetCommunicatorGroup(group_key);
  if (comm_group == nullptr || local_dev_id < 0 || local_dev_id >= device_count_ || peer_dev_id < 0 ||
      peer_dev_id >= device_count_) {
    KLLM_LOG_ERROR << "[NCCL] Communication group not found: " << group_key;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& resource = comm_group->device_resources_[local_dev_id];
  if (!resource || !resource->recv_comm) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Recv communicator not initialized");
  }

  cudaStream_t cur_stream;
  if (stream == nullptr) {
    // 如果没有提供流，则使用默认的设备流
    cur_stream = resource->recv_stream;
  } else {
    cur_stream = stream;
  }
  int send_rank;
  if (local_dev_id == peer_dev_id) {
    send_rank = resource->send_rank;
  } else {
    send_rank = comm_group->device_resources_[peer_dev_id]->send_rank;
  }

  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  NCCL_CHECK(ncclRecv(buf, count, nccl_dtype, send_rank, resource->recv_comm, cur_stream));

  if (!stream) CUDA_CHECK(cudaStreamSynchronize(cur_stream));

  return Status();
}

Status NcclCommunicator::SendGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                                   const std::vector<const void*>& buffers, const std::vector<size_t>& counts,
                                   DataType dtype) {
  KLLM_LOG_DEBUG << "[NCCL] SendGroup starting for group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", buffer_count=" << buffers.size()
                 << ", from node_rank=" << node_rank_;
  if (buffers.empty()) {
    return Status();  // 无数据需要发送，直接返回成功
  }
  // 检查输入参数
  if (buffers.size() != counts.size()) {
    KLLM_LOG_ERROR << "[NCCL] SendGroup parameter mismatch: buffers size " << buffers.size() << " != counts size "
                   << counts.size();
    return Status(RetCode::RET_INVALID_ARGUMENT, "Buffers and counts vectors must have the same size");
  }

  // 获取通信组
  auto comm_group = NcclCommunicator::GetCommunicatorGroup(group_key);
  if (comm_group == nullptr || local_dev_id < 0 || local_dev_id >= device_count_ || peer_dev_id < 0 ||
      peer_dev_id >= device_count_) {
    KLLM_LOG_ERROR << "[NCCL] Communication group not found: " << group_key;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& resource = comm_group->device_resources_[local_dev_id];
  if (!resource || !resource->send_comm) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Send communicator not initialized for SendGroup");
  }

  cudaStream_t cur_stream;
  if (stream == nullptr) {
    // 如果没有提供流，则使用默认的设备流
    cur_stream = resource->send_stream;
  } else {
    cur_stream = stream;
  }
  int recv_rank;
  if (local_dev_id == peer_dev_id) {
    recv_rank = resource->recv_rank;
  } else {
    recv_rank = comm_group->device_resources_[peer_dev_id]->recv_rank;
  }

  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);

  // 使用ncclGroupStart/End优化多次通信操作
  KLLM_LOG_DEBUG << "[NCCL] Starting ncclGroupStart for SendGroup to peer_rank=" << recv_rank;

  NCCL_CHECK(ncclGroupStart());
  // 批量发送所有缓冲区
  size_t total_elements = 0;
  for (size_t i = 0; i < buffers.size(); ++i) {
    NCCL_CHECK(ncclSend(buffers[i], counts[i], nccl_dtype, recv_rank, resource->send_comm, cur_stream));
    total_elements += counts[i];
  }

  // 结束组操作
  NCCL_CHECK(ncclGroupEnd());
  KLLM_LOG_DEBUG << "[NCCL] Completed ncclGroupEnd for SendGroup, waiting for stream synchronize";

  if (!stream) CUDA_CHECK(cudaStreamSynchronize(cur_stream));

  return Status();
}

Status NcclCommunicator::RecvGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                                   const std::vector<void*>& buffers, const std::vector<size_t>& counts,
                                   DataType dtype) {
  KLLM_LOG_DEBUG << "[NCCL] RecvGroup starting for group_key=" << group_key << ", local_dev_id=" << local_dev_id
                 << ", peer_dev_id=" << peer_dev_id << ", buffer_count=" << buffers.size()
                 << ", to node_rank=" << node_rank_;

  // 检查输入参数
  if (buffers.size() != counts.size()) {
    KLLM_LOG_ERROR << "[NCCL] RecvGroup parameter mismatch: buffers size " << buffers.size() << " != counts size "
                   << counts.size();
    return Status(RetCode::RET_INVALID_ARGUMENT, "Buffers and counts vectors must have the same size");
  }

  auto comm_group = NcclCommunicator::GetCommunicatorGroup(group_key);
  if (comm_group == nullptr || local_dev_id < 0 || local_dev_id >= device_count_ || peer_dev_id < 0 ||
      peer_dev_id >= device_count_) {
    KLLM_LOG_ERROR << "[NCCL] Communication group not found: " << group_key;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
  }

  auto& resource = comm_group->device_resources_[local_dev_id];
  if (!resource || !resource->recv_comm) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Recv communicator not initialized for RecvGroup");
  }

  cudaStream_t cur_stream;
  if (stream == nullptr) {
    // 如果没有提供流，则使用默认的设备流
    cur_stream = resource->recv_stream;
  } else {
    cur_stream = stream;
  }
  int send_rank;
  if (local_dev_id == peer_dev_id) {
    send_rank = resource->send_rank;
  } else {
    send_rank = comm_group->device_resources_[peer_dev_id]->send_rank;
  }

  ncclDataType_t nccl_dtype;
  GetNcclDataType(dtype, nccl_dtype);
  // 使用ncclGroupStart/End优化多次通信操作
  KLLM_LOG_DEBUG << "[NCCL] Starting ncclGroupStart for RecvGroup from peer_rank=" << send_rank;

  NCCL_CHECK(ncclGroupStart());
  // 批量接收所有缓冲区
  size_t total_elements = 0;
  for (size_t i = 0; i < buffers.size(); ++i) {
    NCCL_CHECK(ncclRecv(buffers[i], counts[i], nccl_dtype, send_rank, resource->recv_comm, cur_stream));
    total_elements += counts[i];
  }
  NCCL_CHECK(ncclGroupEnd());
  KLLM_LOG_DEBUG << "[NCCL] Completed ncclGroupEnd for RecvGroup, waiting for stream synchronize";

  // 确保组通信操作完成
  if (!stream) CUDA_CHECK(cudaStreamSynchronize(cur_stream));

  return Status();
}

// 获取通信组指针（只读，不转移所有权）
NcclCommGroup* NcclCommunicator::GetCommunicatorGroup(const std::string& group_key) {
  std::lock_guard<std::mutex> lock(comm_group_mutex_);
  auto it = comm_groups_.find(group_key);
  return it != comm_groups_.end() ? it->second.get() : nullptr;
}
#endif  // ENABLE_CUDA
}  // namespace ksana_llm