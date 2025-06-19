/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/data_channel_interface.h"
#include "ksana_llm/distributed/raw_socket.h"

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

#include "ksana_llm/distributed/data_channel.h"

namespace ksana_llm {

// Used to send & recv data message.
class ExpertParallelDataChannel : public DataChannelInterface {
 public:
  ExpertParallelDataChannel(PacketCreationFunc packet_creation_fn = GetPacketObject,
                            HiddenUnitBufferPool* hidden_unit_buffer_pool = nullptr,
                            ExpertParallelHiddenUnitBufferPool* expert_paarallel_hidden_unit_buffer_pool = nullptr,
                            std::shared_ptr<Environment> env = nullptr);
  virtual ~ExpertParallelDataChannel();

  virtual Status Listen() override;

  // Close open port.
  virtual Status Close() override;

  // For normal node only.
  virtual Status Connect() override;

  // disconnect from master.
  virtual Status Disconnect() override;

 private:
  // Invoked when data arrives.
  Status HandleServerPacket(NodeInfo* node_info, Packet* packet);
  Status HandleClientPacket(NodeInfo* node_info, Packet* packet);

  // Process hidden units.
  Status ProcessHiddenUnitRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessHiddenUnitResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Send data to downstream node.
  Status ProcessSendPacketLoop();

  // Copy received packet to device if device buffer is free.
  Status ProcessHostToDeviceLoop();

 private:
  std::shared_ptr<RawSocket> server_raw_socket_ = nullptr;
  std::shared_ptr<RawSocket> client_raw_socket_ = nullptr;

  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  // The buffer pool.
  ExpertParallelHiddenUnitBufferPool* hidden_unit_buffer_pool_ = nullptr;

  // Used to copy host memory to device buffer, and vice versa
  std::shared_ptr<std::thread> host_to_device_thread_ = nullptr;

  // Send data buffers to remote.
  std::shared_ptr<std::thread> send_packet_thread_ = nullptr;

  // data ip and port.
  std::string data_host_;
  uint16_t data_port_;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
