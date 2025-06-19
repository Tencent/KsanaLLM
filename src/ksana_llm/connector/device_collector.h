/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

// 系统头文件
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/utils/socket_util.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif

namespace ksana_llm {

// 设备信息收集类
class DeviceCollector {
 public:
#ifdef ENABLE_CUDA
  // 收集NVIDIA GPU设备信息
  static void CollectGPUDevices(int device_count, std::vector<DeviceInfo>& devices) {
    std::string devicd_ip;
    std::string interface;
    // TODO(shawnding):  Get Real Device IP
    GetAvailableInterfaceAndIP(interface, devicd_ip);
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::string type = std::string(prop.name);
      devices.emplace_back(i, type, devicd_ip);
    }
  }
#endif

#ifdef ENABLE_ACL
  // 收集Ascend NPU设备信息
  static void CollectNPUDevices(int device_count, std::vector<DeviceInfo>& devices) {
    std::string devicd_ip;
    std::string interface;
    // TODO(shawnding):  Get Real Device IP
    GetAvailableInterfaceAndIP(interface, devicd_ip);
    for (int i = 0; i < device_count; ++i) {
      std::string type = "Ascend NPU";
      devices.emplace_back(i, type, devicd_ip);
    }
  }
#endif

  static std::vector<DeviceInfo> CollectDeviceInformation(int device_count, const std::string& host_ip) {
    std::vector<DeviceInfo> devices;
#ifdef ENABLE_CUDA
    CollectGPUDevices(device_count, devices);
#endif
#ifdef ENABLE_ACL
    CollectNPUDevices(device_count, devices);
#endif
    if (devices.empty()) {
      devices.emplace_back(0, "CPU", host_ip);
    }
    return devices;
  }
};

}  // namespace ksana_llm
