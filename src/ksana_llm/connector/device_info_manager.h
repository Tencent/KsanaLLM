/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ksana_llm {

/**
 * @brief 存储 Decode 节点的DP组和总设备数信息
 *
 * 对于Decode节点，负责保存和管理该节点的DP组和总设备数信息，
 * 避免设备信息重复传输。
 * 对于Prefill节点，负责接收通过ZMQ传输的Decode节点的DP组和总设备数信息，
 * 以便在Prefill节点上正确地分配transfer task。
 */

class DeviceInfoManager {
 public:
  DeviceInfoManager() = default;
  ~DeviceInfoManager() = default;

  DeviceInfoManager(const DeviceInfoManager&) = delete;
  DeviceInfoManager& operator=(const DeviceInfoManager&) = delete;

  DeviceInfoManager(DeviceInfoManager&&) = default;
  DeviceInfoManager& operator=(DeviceInfoManager&&) = default;

  void Insert(const std::string& group_key, int adp_num, int dev_total_num);

  bool Find(const std::string& group_key, std::pair<int, int>& device_info_pair) const;

  bool FindAndInsert(const std::string& group_key, int adp_num, int dev_total_num);

  void WaitFor(const std::string& group_key, std::pair<int, int>& device_info_pair);

  bool TryGet(const std::string& group_key, std::pair<int, int>& device_info_pair) const;

 private:
  // key 为 kv_comm_group_key, value 为 <decode_dp_num, decode_total_device_num>
  std::unordered_map<std::string, std::pair<int, int>> decode_dev_config_map_;
  mutable std::mutex map_mutex_;
  mutable std::condition_variable cv_;
};

}  // namespace ksana_llm
