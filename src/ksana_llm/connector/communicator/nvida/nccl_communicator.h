/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "ksana_llm/connector/communicator/communicator.h"
#include "ksana_llm/utils/logger.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#  include "ksana_llm/utils/nvidia/nccl_utils.h"
#endif
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
#ifdef ENABLE_CUDA
struct NcclDeviceResource {
  /** @brief CUDA stream for send operations */
  cudaStream_t send_stream = nullptr;

  /** @brief NCCL communicator for send operations */
  ncclComm_t send_comm = nullptr;

  /** @brief CUDA stream for receive operations */
  cudaStream_t recv_stream = nullptr;

  /** @brief NCCL communicator for receive operations */
  ncclComm_t recv_comm = nullptr;

  /** @brief CUDA stream for memory copy operations */
  cudaStream_t memcpy_stream = nullptr;

  /** @brief Send rank in NCCL communication */
  int send_rank = -1;

  /** @brief Receive rank in NCCL communication */
  int recv_rank = -1;

  /**
   * @brief Default constructor
   */
  NcclDeviceResource() = default;

  /**
   * @brief Destructor to clean up resources
   */
  ~NcclDeviceResource() = default;
};

struct NcclCommGroup {
  /** @brief List of communication ID strings */
  std::string comm_id_str;

  /** @brief NCCL unique ID for establishing communication */
  ncclUniqueId comm_id;
  std::vector<std::tuple<int, int, std::string>> address_tuples;

  std::vector<std::unique_ptr<NcclDeviceResource>> device_resources_;

  /** @brief Number of devices in this communication group */
  int device_count_ = 0;

  NcclCommGroup() = default;
  ~NcclCommGroup() = default;

  // 禁止拷贝
  NcclCommGroup(const NcclCommGroup&) = delete;
  NcclCommGroup& operator=(const NcclCommGroup&) = delete;

  NcclDeviceResource* GetDeviceResource(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(device_resources_.size())) return nullptr;
    return device_resources_[device_id].get();
  }

  bool IsResourceValid(const NcclDeviceResource& resource) const {
    return resource.send_comm && resource.recv_comm && resource.send_stream && resource.recv_stream;
  }

  bool IsDeviceResourceValid(int device_id) const {
    if (device_id < 0 || device_id >= static_cast<int>(device_resources_.size())) return false;
    const auto* resource = device_resources_[device_id].get();
    return resource && IsResourceValid(*resource);
  }

  bool IsActive() const {
    // 如果没有设备资源，认为不活跃
    if (device_resources_.empty()) return false;
    // 检查所有设备资源是否都有效
    for (const auto& resource : device_resources_) {
      if (!resource || !IsResourceValid(*resource)) {
        return false;
      }
    }
    return true;
  }
};

class NcclCommunicator : public Communicator {
 public:
  explicit NcclCommunicator(const ksana_llm::ConnectorConfig& config) : Communicator(config) {}
  ~NcclCommunicator() override;

  void Shutdown() override;

  Status Initialize() override;

  Status ProcessHeartbeatData(const std::unordered_map<std::string, std::string>& comm_group_to_id,
                              const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&
                                  comm_group_to_address) override;

  std::unique_ptr<NcclDeviceResource> CreateNcclDeviceResource(const ConnectorConfig& config,
                                                               const ncclUniqueId& comm_id, int node_rank,
                                                               int world_size, int device_count, int dev_id);

  // Communicator 接口实现
  Status Send(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, const void* buf,
              size_t count, DataType dtype) override;
  Status Recv(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id, void* buf, size_t count,
              DataType dtype) override;
  Status SendGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<const void*>& buffers, const std::vector<size_t>& counts, DataType dtype) override;
  Status RecvGroup(const std::string& group_key, int src_dev_id, int dst_dev_id, uint64_t job_id,
                   const std::vector<void*>& buffers, const std::vector<size_t>& counts, DataType dtype) override;

  // Communicator 接口实现
  Status Send(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, const void* buf,
              size_t count, DataType dtype);
  Status Recv(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, void* buf,
              size_t count, DataType dtype);
  Status SendGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                   const std::vector<const void*>& buffers, const std::vector<size_t>& counts, DataType dtype);
  Status RecvGroup(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream,
                   const std::vector<void*>& buffers, const std::vector<size_t>& counts, DataType dtype);

  struct NcclRankInfo {
    int cur_rank;
    int peer_rank;
    int send_rank;
    int recv_rank;
  };
  NcclRankInfo CalcNcclRanks(GroupRole group_role, int node_rank, int world_size, int device_count, int dev_id);

  // 在 NcclCommunicator 增加测试专用 comm_group 插入方法
  void SetCommGroupForTest(const std::string& group_key, std::unique_ptr<NcclCommGroup> group) {
    std::lock_guard<std::mutex> lock(comm_group_mutex_);
    comm_groups_[group_key] = std::move(group);
  }

  void SetSendCommIdCallback(std::function<Status(const std::string&, const std::string&)> callback) {
    send_comm_id_callback_ = std::move(callback);
  }

  Status CreateDeviceResources(const std::string& group_key);

  /**
   * @brief Protected implementation of SetReceiveCallback (implements Communicator interface)
   *
   * @param callback Function to call when data is received
   */
  void DoSetReceiveCallback(const ReceiveCallback& callback) override;

  bool CheckCommGroupAvailable(const std::string& group_key);

  void CreateNcclCommGroup(const std::string& group_key, const std::string& encoded_value);
  bool IsConnectionReady(const std::string& group_key, int dev_id) const;
  NcclCommGroup* GetCommunicatorGroup(const std::string& group_key);

 private:
  std::mutex comm_group_mutex_;
  std::unordered_map<std::string, std::unique_ptr<NcclCommGroup>> comm_groups_;

  // 回调函数，用于发送通信ID到协调器
  std::function<Status(const std::string&, const std::string&)> send_comm_id_callback_;

  /** @brief Callback for received messages (modern C++ style) */
  ReceiveCallback receive_callback_;
  void* recv_user_data_ = nullptr;
};

#endif  // ENABLE_CUDA
}  // namespace ksana_llm