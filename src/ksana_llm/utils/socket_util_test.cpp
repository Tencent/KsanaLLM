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

  struct ifaddrs* if_addr = nullptr;
  getifaddrs(&if_addr);
  for (struct ifaddrs* ifa = if_addr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) {
      continue;
    }

    if (ifa->ifa_addr->sa_family == AF_INET && (ifa->ifa_flags & IFF_LOOPBACK) == 0) {
      char address_buffer[INET_ADDRSTRLEN];
      void* sin_addr_ptr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr;
      inet_ntop(AF_INET, sin_addr_ptr, address_buffer, INET_ADDRSTRLEN);
      ip = address_buffer;
      interface = ifa->ifa_name;

      break;
    }
  }
  if (nullptr != if_addr) {
    freeifaddrs(if_addr);
  }

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
