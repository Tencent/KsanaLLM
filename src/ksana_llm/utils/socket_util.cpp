/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status GetAvailableInterfaceAndIP(std::string& interface, std::string& ip) {
  interface.clear();
  ip.clear();
  std::unordered_map<std::string, std::string> interface_ip;

  ifaddrs* if_addr = nullptr;
  getifaddrs(&if_addr);
  for (struct ifaddrs* ifa = if_addr; ifa != nullptr; ifa = ifa->ifa_next) {
    // 跳过无效、非IPv4、回环和docker开头的接口
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET || (ifa->ifa_flags & IFF_LOOPBACK) != 0) {
      continue;
    }
    if (ifa->ifa_name && strncmp(ifa->ifa_name, "docker", 6) == 0) {
      continue;
    }

    char address_buffer[INET_ADDRSTRLEN];
    void* const sin_addr_ptr = &(reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr);
    inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);

    interface_ip[std::string(ifa->ifa_name)] = std::string(address_buffer);
  }

  if (nullptr != if_addr) {
    freeifaddrs(if_addr);
  }

  if (interface_ip.empty()) {
    KLLM_LOG_ERROR << "error ip info";
    return Status();
  }

  const char* const ifa_name_env = std::getenv("IFA_NAME");
  const std::string ifa_name = ifa_name_env == nullptr ? "bond1" : std::string(ifa_name_env);

  auto interface_ip_it = interface_ip.find(ifa_name);
  if (interface_ip_it == interface_ip.end()) {
    interface_ip_it = interface_ip.begin();
  }

  interface = interface_ip_it->first;
  ip = interface_ip_it->second;

  return Status();
}

Status GetAvailablePort(uint16_t& port) {
  // Pick up a random port available for me
  struct sockaddr_in addr;
  addr.sin_port = htons(0);
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    return Status(RET_BIND_PORT_FAILED, "Get available port error, bind failed.");
  }

  socklen_t addr_len = sizeof(struct sockaddr_in);
  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    return Status(RET_BIND_PORT_FAILED, "Get available port error, getsockname failed.");
  }

  port = ntohs(addr.sin_port);

  close(sock);
  return Status();
}

}  // namespace ksana_llm
