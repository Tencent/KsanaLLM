/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/raw_socket.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/node_info.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/service_utils.h"
#include "ksana_llm/utils/status.h"

#include "fmt/core.h"

namespace ksana_llm {

RawSocket::RawSocket(PacketCreationFunc packet_creation_fn) { packet_creation_fn_ = packet_creation_fn; }

Status RawSocket::ExtractHostPort(struct sockaddr_in* sockaddr, std::string& host, uint16_t& port) {
  char host_cstr[INET_ADDRSTRLEN];

  inet_ntop(AF_INET, &sockaddr->sin_addr, host_cstr, sizeof(host_cstr));
  port = htons(sockaddr->sin_port);
  host = host_cstr;

  return Status();
}

void RawSocket::DropPacket(int socket_fd) {
  char buffer[1024];
  while (true) {
    if (read(socket_fd, buffer, sizeof(buffer)) <= 0) {
      break;
    }
  }
}

bool RawSocket::CheckSocketReadable(int socket_fd, int timeout_sec) {
  fd_set read_fds;
  FD_ZERO(&read_fds);
  FD_SET(socket_fd, &read_fds);

  struct timeval timeout;
  timeout.tv_sec = timeout_sec;
  timeout.tv_usec = 0;

  int select_ret;
  do {
    select_ret = select(socket_fd + 1, &read_fds, NULL, NULL, &timeout);
  } while (select_ret < 0 && errno == EINTR);

  // error or timeout, no data
  if (select_ret == -1 || select_ret == 0) {
    return false;
  }

  return true;
}

void RawSocket::RecvPacket(PacketHandle& packet_handle, std::vector<Packet*>& packets, size_t& recv_bytes) {
  int& socket_fd = packet_handle.socket_fd;
  PacketBuffer& packet_buffer = packet_handle.packet_buffer;

  // magic number and packet size and packet type.
  constexpr int leading_size = 2 * sizeof(int) + sizeof(PacketType);

  recv_bytes = 0;
  while (!packet_handle.terminated) {
    if (packet_buffer.pending_state == PacketPendingState::PENDING_NONE) {
      packet_buffer.head_recv_offset = 0;
      packet_buffer.head_left_bytes = leading_size;
      packet_buffer.pending_state = PacketPendingState::PENDING_HEAD;
    }

    if (packet_buffer.pending_state == PacketPendingState::PENDING_HEAD) {
      if (!CheckSocketReadable(socket_fd, 1)) {
        continue;
      }

      // Read leading bytes.
      int bytes_read =
          read(socket_fd, packet_buffer.pending_head + packet_buffer.head_recv_offset, packet_buffer.head_left_bytes);
      if (bytes_read <= 0) {
        // error or EOF, close socket and stop recv thread.
        close(socket_fd);
        packet_handle.terminated = true;

        KLLM_LOG_WARNING << "Read packet head from socket " << socket_fd << " error, info: " << strerror(errno)
                         << ", head recv offset:" << packet_buffer.head_recv_offset
                         << ", head left bytes:" << packet_buffer.head_left_bytes << ", bytes:" << bytes_read
                         << ", close it.";

        // stop node if socket disconnected.
        if (bytes_read == 0) {
          std::thread([]() -> void { GetServiceLifetimeManager()->ShutdownService(); }).detach();
        }
        break;
      }

      recv_bytes += bytes_read;
      packet_buffer.head_recv_offset += bytes_read;
      packet_buffer.head_left_bytes -= bytes_read;

      if (packet_buffer.head_left_bytes > 0) {
        break;
      }

      // Check magic number.
      int magic_number = *reinterpret_cast<int*>(packet_buffer.pending_head);
      int packet_size = *reinterpret_cast<int*>(packet_buffer.pending_head + sizeof(int));
      if (magic_number != PACKET_MAGIC_NUMBER || packet_size <= 0) {
        DropPacket(socket_fd);
        packet_buffer.pending_state = PacketPendingState::PENDING_NONE;
        packet_buffer.pending_packet = nullptr;

        KLLM_LOG_ERROR << "Receive unknown packet format from packet " << socket_fd;
        break;
      }

      // Get packet object from type.
      PacketType packet_type = *reinterpret_cast<PacketType*>(packet_buffer.pending_head + (2 * sizeof(int)));
      Packet* packet = packet_creation_fn_(packet_type, packet_size);
      if (packet == nullptr) {
        throw std::runtime_error("RawSocket::RecvPacket allocate memory error.");
      }

      // Ready to recv packet.
      packet_buffer.pending_packet = packet;
      packet_buffer.packet_left_bytes = packet_size - sizeof(PacketType);
      packet_buffer.packet_recv_offset = sizeof(PacketType);
      packet_buffer.pending_state = PacketPendingState::PENDING_BODY;
    }

    if (packet_buffer.pending_state == PacketPendingState::PENDING_BODY) {
      if (!CheckSocketReadable(socket_fd, 1)) {
        continue;
      }

      int bytes_read =
          read(socket_fd, reinterpret_cast<char*>(packet_buffer.pending_packet) + packet_buffer.packet_recv_offset,
               packet_buffer.packet_left_bytes);
      if (bytes_read <= 0) {
        // error or EOF, close the socket.
        close(socket_fd);
        packet_handle.terminated = true;

        KLLM_LOG_WARNING << "Read packet body from socket " << socket_fd << " error, info: " << strerror(errno)
                         << ", packet buffer size:" << packet_buffer.pending_packet->size
                         << ", packet recv offset:" << packet_buffer.packet_recv_offset
                         << ", packet left bytes:" << packet_buffer.packet_left_bytes << ", read bytes:" << bytes_read
                         << ", close it.";

        // stop node if socket disconnected.
        if (bytes_read == 0) {
          std::thread([]() -> void { GetServiceLifetimeManager()->ShutdownService(); }).detach();
        }
        break;
      }

      recv_bytes += bytes_read;
      packet_buffer.packet_recv_offset += bytes_read;
      packet_buffer.packet_left_bytes -= bytes_read;

      if (packet_buffer.packet_left_bytes > 0) {
        break;
      }

      // Packet finished.
      packets.push_back(packet_buffer.pending_packet);

      packet_buffer.pending_packet = nullptr;
      packet_buffer.pending_state = PacketPendingState::PENDING_NONE;

      // Return if got a complete packet.
      break;
    }
  }
}

Status RawSocket::SendPacket(int socket_fd, const Packet* packet) {
  int magic_number = PACKET_MAGIC_NUMBER;
  if (write(socket_fd, &magic_number, sizeof(int)) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Send magic number on socket {} error.", socket_fd));
  }

  int packet_size = sizeof(Packet) + packet->size;
  if (write(socket_fd, &packet_size, sizeof(int)) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Send packet size on socket {} error.", socket_fd));
  }

  if (write(socket_fd, packet, packet_size) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Send packet data on socket {} error.", socket_fd));
  }

  return Status();
}

void RawSocket::InitPacketHandle(PacketHandle& packet_handle) {
  auto thread_fn = [this, &packet_handle]() -> void {
    while (!packet_handle.terminated) {
      // Recv until no data recv.
      size_t recv_bytes;
      std::vector<Packet*> packets;
      do {
        RecvPacket(packet_handle, packets, recv_bytes);
      } while (packets.empty() && (!packet_handle.terminated));

      for (Packet* packet : packets) {
        Status status = packet_handle_cb_(&packet_handle.node_info, packet);
        if (!status.OK()) {
          KLLM_LOG_ERROR << "Invoke packet callback error:" << status.GetMessage();
        }
      }
    }
  };

  packet_handle.recv_thread = std::make_shared<std::thread>(thread_fn);
}

void RawSocket::AddPacketHandle(const NodeInfo& node_info, int socket_fd) {
  KLLM_LOG_INFO << "Add packet handle for " << node_info.host << ":" << node_info.port << ", fd:" << socket_fd;

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = node_fd_.find(node_info);
  if (it != node_fd_.end()) {
    throw std::runtime_error(fmt::format("Duplicate node info {}:{}", node_info.host, node_info.port));
  }
  node_fd_[node_info] = socket_fd;

  auto it2 = fd_packet_handle_.find(socket_fd);
  if (it2 != fd_packet_handle_.end()) {
    throw std::runtime_error(fmt::format("Duplicate packet handle of fd {}", socket_fd));
  }

  PacketHandle packet_handle;
  packet_handle.socket_fd = socket_fd;
  packet_handle.node_info = node_info;
  fd_packet_handle_[socket_fd] = packet_handle;

  // Start handle thread for this fd.
  InitPacketHandle(fd_packet_handle_[socket_fd]);
}

void RawSocket::StopPacketHandle(PacketHandle& packet_handle) {
  KLLM_LOG_INFO << "Stop packet handle " << packet_handle.socket_fd;

  // Try to close the fd, even if it have been closed.
  close(packet_handle.socket_fd);

  // Set terminated flag if not set yet.
  packet_handle.terminated = true;
  if (packet_handle.recv_thread) {
    KLLM_LOG_INFO << "Stop packet handle " << packet_handle.socket_fd << ", join thread";
    packet_handle.recv_thread->join();
    KLLM_LOG_INFO << "Stop packet handle " << packet_handle.socket_fd << ", join thread finish.";
  }
}

void RawSocket::DelPacketHandle(int socket_fd) {
  KLLM_LOG_INFO << "Remove packet handle for fd " << socket_fd;

  std::lock_guard<std::mutex> lock(mutex_);

  if (fd_packet_handle_.find(socket_fd) != fd_packet_handle_.end()) {
    PacketHandle& packet_handle = fd_packet_handle_[socket_fd];

    if (node_fd_.find(packet_handle.node_info) != node_fd_.end()) {
      node_fd_.erase(packet_handle.node_info);
    }

    StopPacketHandle(packet_handle);
    fd_packet_handle_.erase(socket_fd);
  }
}

Status RawSocket::Listen(const std::string& host, uint16_t port, PacketProcessFunc cb) {
  packet_handle_cb_ = cb;

  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ == -1) {
    return Status(RET_RUNTIME_FAILED, "Create socket error.");
  }

  struct sockaddr_in srv_address;
  srv_address.sin_family = AF_INET;
  srv_address.sin_addr.s_addr = inet_addr(host.c_str());
  srv_address.sin_port = htons(port);
  if (bind(server_fd_, (struct sockaddr*)&srv_address, sizeof(srv_address)) == -1) {
    close(server_fd_);
    return Status(RET_RUNTIME_FAILED, fmt::format("Bind socket {}:{} error.", host, port));
  }

  if (fcntl(server_fd_, F_SETFL, fcntl(server_fd_, F_GETFL, 0) | O_NONBLOCK) == -1) {
    close(server_fd_);
    return Status(RET_RUNTIME_FAILED, "Set nonblocking mode error.");
  }

  constexpr int max_connctions = 1024;
  if (listen(server_fd_, max_connctions) == -1) {
    close(server_fd_);
    return Status(RET_RUNTIME_FAILED, "Listen socket error.");
  }

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ == -1) {
    close(server_fd_);
    return Status(RET_RUNTIME_FAILED, "Create epoll instance error.");
  }

  struct epoll_event event;
  event.events = EPOLLIN | EPOLLET;
  event.data.fd = server_fd_;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, server_fd_, &event) == -1) {
    close(server_fd_);
    close(epoll_fd_);
    return Status(RET_RUNTIME_FAILED, "Failed to add server socket to epoll instance.");
  }

  auto thread_fn = [this]() -> void {
    constexpr int max_event = 1024;
    struct epoll_event events[max_event];

    while (!terminated_) {
      int num_events = epoll_wait(epoll_fd_, events, max_event, 1000);
      if (num_events == -1) {
        KLLM_LOG_ERROR << "RawSocket::Listen Wait for epoll event error.";
        continue;
      } else if (num_events == 0) {
        // timeout 1000ms
        continue;
      }

      for (int i = 0; i < num_events; ++i) {
        // new client connection.
        if (events[i].data.fd == server_fd_) {
          struct sockaddr_in client_address;
          socklen_t client_addr_len = sizeof(client_address);

          // The client_fd is in blocking mode in default.
          int client_fd = accept(server_fd_, (struct sockaddr*)&client_address, &client_addr_len);
          if (client_fd == -1) {
            throw std::runtime_error("Accept client connection error.");
          }

          NodeInfo node_info;
          ExtractHostPort(&client_address, node_info.host, node_info.port);

          // Refuse any new connection if forzen.
          if (is_frozen_) {
            KLLM_LOG_WARNING << "New connect from " << node_info.host << ":" << node_info.port
                             << " is rejected,  because of the socket is frozen.";
            close(client_fd);
            continue;
          }

          // Add a new handler.
          AddPacketHandle(node_info, client_fd);
        }
      }
    }
  };

  server_thread_ = std::make_shared<std::thread>(thread_fn);
  is_connected_ = true;

  return Status();
}

Status RawSocket::Close() {
  is_connected_ = false;
  terminated_ = true;

  // Close fd first, otherwise the read() will be blocked.
  close(epoll_fd_);
  close(server_fd_);

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Waiting all packet handler finished.
    for (auto& pair : fd_packet_handle_) {
      PacketHandle& packet_handle = pair.second;
      StopPacketHandle(packet_handle);
    }
    fd_packet_handle_.clear();
    node_fd_.clear();
  }

  if (server_thread_) {
    server_thread_->join();
  }

  return Status();
}

Status RawSocket::Connect(const std::string& host, uint16_t port, PacketProcessFunc cb) {
  if (is_frozen_) {
    KLLM_LOG_WARNING << "Connect to " << host << ":" << port << " error, the socket is frozen.";
    return Status(RET_RUNTIME_FAILED, "Could not connect, the socket is frozen yet.");
  }

  packet_handle_cb_ = cb;

  // The client_fd_ is in blocking mode in default.
  client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd_ == -1) {
    return Status(RET_RUNTIME_FAILED, "Create socket error.");
  }

  struct sockaddr_in srv_addr;
  srv_addr.sin_family = AF_INET;
  srv_addr.sin_addr.s_addr = inet_addr(host.c_str());
  srv_addr.sin_port = htons(port);
  if (connect(client_fd_, (struct sockaddr*)&srv_addr, sizeof(srv_addr)) == -1) {
    close(client_fd_);
    return Status(RET_RUNTIME_FAILED, fmt::format("Connect socket {}:{} error.", host, port));
  }

  NodeInfo node_info;
  node_info.host = host;
  node_info.port = port;

  AddPacketHandle(node_info, client_fd_);
  is_connected_ = true;

  return Status();
}

Status RawSocket::Disconnect() {
  is_connected_ = false;
  DelPacketHandle(client_fd_);
  return Status();
}

Status RawSocket::Frozen() {
  KLLM_LOG_INFO << "Set socket to frozen, any new connection will be rejected.";
  is_frozen_ = true;
  return Status();
}

Status RawSocket::Send(NodeInfo node_info, const Packet* packet) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = node_fd_.find(node_info);
  if (it == node_fd_.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Node {}:{} not found.", node_info.host, node_info.port));
  }

  return SendPacket(node_fd_[node_info], packet);
}

bool RawSocket::IsConnected() { return is_connected_; }

}  // namespace ksana_llm
