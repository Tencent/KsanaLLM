/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include "ksana_llm/connector/device_info_manager.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// 主要用在Prefill端, 接收Decode端传输的设备信息后插入到map中
void DeviceInfoManager::Insert(const std::string& group_key, int adp_num, int dev_total_num) {
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    decode_dev_config_map_[group_key] = std::make_pair(adp_num, dev_total_num);
  }
  cv_.notify_all();
}

bool DeviceInfoManager::Find(const std::string& group_key, std::pair<int, int>& device_info_pair) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = decode_dev_config_map_.find(group_key);
  if (it != decode_dev_config_map_.end()) {
    device_info_pair = it->second;
    return true;
  }
  return false;
}

// 主要用在Decode端, 便于确认设备信息是否已经传输
bool DeviceInfoManager::FindAndInsert(const std::string& group_key, int adp_num, int dev_total_num) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = decode_dev_config_map_.find(group_key);
  if (it == decode_dev_config_map_.end()) {
    decode_dev_config_map_[group_key] = std::make_pair(adp_num, dev_total_num);
    return false;
  }
  return true;
}

// 主要用在Prefill端, 等待Decode端传输设备信息
void DeviceInfoManager::WaitFor(const std::string& group_key, std::pair<int, int>& device_info_pair) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  cv_.wait(lock,
           [this, &group_key]() { return decode_dev_config_map_.find(group_key) != decode_dev_config_map_.end(); });
  device_info_pair = decode_dev_config_map_[group_key];
}

// 主要用在Prefill端, 尝试获取设备信息
bool DeviceInfoManager::TryGet(const std::string& group_key, std::pair<int, int>& device_info_pair) const {
  return Find(group_key, device_info_pair);
}

}  // namespace ksana_llm