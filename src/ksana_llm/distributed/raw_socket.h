/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <netinet/in.h>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "ksana_llm/distributed/node_info.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

enum PacketPendingState { PENDING_NONE, PENDING_HEAD, PENDING_BODY };

// Used to process pending packet.
struct PacketBuffer {
  char pending_head[12];
  size_t head_left_bytes = 0;
  size_t head_recv_offset = 0;

  Packet* pending_packet = nullptr;
  size_t packet_left_bytes = 0;
  size_t packet_recv_offset = 0;

  PacketPendingState pending_state = PacketPendingState::PENDING_NONE;
};

struct PacketHandle {
  int socket_fd;
  NodeInfo node_info;
  std::shared_ptr<std::thread> recv_thread = nullptr;
  bool terminated = false;

  PacketBuffer packet_buffer;
};

// A socket ipc implementation using epoll.
// Using a std::thread to process received packet, the handler func is passed in through Listen() or Connect().
class RawSocket {
 public:
  explicit RawSocket(PacketCreationFunc packet_creation_fn);

  // For master node only.
  Status Listen(const std::string& host, uint16_t port, PacketProcessFunc cb);

  // Close open port.
  Status Close();

  // For normal node only.
  Status Connect(const std::string& host, uint16_t port, PacketProcessFunc cb);

  // disconnect from master.
  Status Disconnect();

  // Stop to accept any new connection.
  Status Frozen();

  // Send data to remote, maybe from device directly.
  Status Send(NodeInfo node_info, const Packet* packet);

  // Whether the socket is connected with remote peer.
  bool IsConnected();

 private:
  Status ExtractHostPort(struct sockaddr_in* sockaddr, std::string& host, uint16_t& port);

  // Initialize a socket handle.
  void InitPacketHandle(PacketHandle& packet_handle);

  // Stop a socket handle.
  void StopPacketHandle(PacketHandle& packet_handle);

  // Add and del packet handle.
  void AddPacketHandle(const NodeInfo& node_info, int socket_fd);
  void DelPacketHandle(int socket_fd);

  // Send packet on socket.
  Status SendPacket(int socket_fd, const Packet* packet);

  // Receive one or more packets from handle.
  void RecvPacket(PacketHandle& packet_handle, std::vector<Packet*>& packets, size_t& recv_bytes);

  // Drop all following data from socket fd.
  void DropPacket(int socket_fd);

  // Check whether a socket is readable.
  bool CheckSocketReadable(int socket_fd, int timeout_sec);

 private:
  // For both server and client.
  int epoll_fd_;

  // For server
  int server_fd_;
  std::shared_ptr<std::thread> server_thread_ = nullptr;

  // For client
  int client_fd_;

  bool terminated_ = false;

  PacketCreationFunc packet_creation_fn_;
  PacketProcessFunc packet_handle_cb_;

  std::unordered_map<int, PacketHandle> fd_packet_handle_;
  std::unordered_map<NodeInfo, int, NodeInfoHash, NodeInfoEqual> node_fd_;

  // Protect packet handles.
  std::mutex mutex_;

  bool is_connected_ = false;
  bool is_frozen_ = false;
};

}  // namespace ksana_llm
