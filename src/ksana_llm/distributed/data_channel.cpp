/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/distributed/data_channel.h"

#include <stdexcept>
#include "fmt/core.h"

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

DataChannel::DataChannel(PacketCreationFunc packet_creation_fn, HiddenUnitBufferPool* hidden_unit_buffer_pool,
                         std::shared_ptr<Environment> env) {
  server_raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);
  client_raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);

  env_ = env ? env : Singleton<Environment>::GetInstance();
  hidden_unit_buffer_pool_ = hidden_unit_buffer_pool ? hidden_unit_buffer_pool : GetHiddenUnitBufferPool();

  // Start fetch batch input thread.
  host_to_device_thread_ = std::unique_ptr<std::thread>(new std::thread(&DataChannel::ProcessHostToDeviceLoop, this));

  // Start packet send thread.
  send_packet_thread_ = std::unique_ptr<std::thread>(new std::thread(&DataChannel::ProcessSendPacketLoop, this));
}

DataChannel::~DataChannel() {
  terminated_ = true;
  hidden_unit_buffer_pool_->Stop();

  if (host_to_device_thread_) {
    host_to_device_thread_->join();
  }

  if (send_packet_thread_) {
    send_packet_thread_->join();
  }
}

Status DataChannel::Listen() {
  std::string interface;
  Status status = GetAvailableInterfaceAndIP(interface, data_host_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data ip error: {}", status.GetMessage()));
  }

  status = GetAvailablePort(data_port_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data port error: {}", status.GetMessage()));
  }

  // Write to environment config
  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  pipeline_config.data_host = data_host_;
  pipeline_config.data_port = data_port_;
  env_->SetPipelineConfig(pipeline_config);

  // Listen data port.
  KLLM_LOG_INFO << "DataChannel Listen on " << data_host_ << ":" << data_port_;
  auto listen_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleServerPacket(node_info, packet);
  };

  status = server_raw_socket_->Listen(data_host_, data_port_, listen_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Listen data channel error:" << status.GetMessage();
  }

  return status;
}

Status DataChannel::Close() {
  terminated_ = true;
  return server_raw_socket_->Close();
}

Status DataChannel::Connect() {
  auto connect_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleClientPacket(node_info, packet);
  };

  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  std::string downstream_host = pipeline_config.downstream_host;
  uint16_t downstream_port = pipeline_config.downstream_port;

  KLLM_LOG_INFO << "DataChannel connect to " << downstream_host << ":" << downstream_port;
  Status status = client_raw_socket_->Connect(downstream_host, downstream_port, connect_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "DataChannel connect error:" << status.GetMessage();
  }

  return status;
}

Status DataChannel::Disconnect() { return client_raw_socket_->Disconnect(); }

Status DataChannel::ProcessHiddenUnitRequest(NodeInfo* node_info, Packet* req_packet) {
  // Add to recv queue.
  return hidden_unit_buffer_pool_->PutToHostRecvQueue(req_packet);
}

Status DataChannel::ProcessHiddenUnitResponse(NodeInfo* node_info, Packet* rsp_packet) {
  // skip now
  return Status();
}

Status DataChannel::HandleServerPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::DATA_REQ_HIDDEN_UNIT: {
      return ProcessHiddenUnitRequest(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status DataChannel::HandleClientPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::DATA_RSP_HIDDEN_UNIT: {
      return ProcessHiddenUnitResponse(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status DataChannel::ProcessHostToDeviceLoop() {
  while (!terminated_) {
    // Wait util recv invoked.
    hidden_unit_buffer_pool_->WaitUtilReadyToRecv();

    // Waiting host buffer.
    Packet* packet = hidden_unit_buffer_pool_->GetFromHostRecvQueue();
    if (!packet) {
      KLLM_LOG_WARNING << "ProcessHostToDeviceLoop empty packet from host send queue, break..";
      break;
    }

    // Waiting usable device buffer
    HiddenUnitDeviceBuffer* hidden_unit_dev = hidden_unit_buffer_pool_->GetDeviceBuffer();
    if (!hidden_unit_dev) {
      KLLM_LOG_WARNING << "ProcessHostToDeviceLoop empty packet from host send queue, break..";
      break;
    }

    HiddenUnitHostBuffer* hidden_unit_host = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

    hidden_unit_buffer_pool_->ConvertHostBufferToDevice(hidden_unit_dev, hidden_unit_host);
    KLLM_LOG_DEBUG << "DataChannel::ProcessHostToDeviceLoop. schedule_id=" << hidden_unit_dev->schedule_id
                   << ", hidden_unit_dev=" << hidden_unit_dev;
    hidden_unit_buffer_pool_->PutToDeviceRecvQueue(hidden_unit_dev);

    // Free host packet.
    hidden_unit_buffer_pool_->FreeHostBuffer(packet);
  }

  return Status();
}

Status DataChannel::ProcessSendPacketLoop() {
  while (!terminated_) {
    // Blocked, waiting util packet is ready.
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetFromSendQueue();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessSendPacketLoop empty hidden_unit from device send queue, break..";
      break;
    }

    // Pick a host buffer.
    Packet* packet = hidden_unit_buffer_pool_->GetHostBuffer();

    // Convert device buffer to host.
    HiddenUnitHostBuffer* hidden_unit_host = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
    hidden_unit_buffer_pool_->ConvertDeviceBufferToHost(hidden_unit_host, hidden_unit);

    // Reset packet size.
    packet->size = hidden_unit_buffer_pool_->GetHostPacketSize(packet);

    // Note: Should get config after its value is updated.
    PipelineConfig pipeline_config;
    env_->GetPipelineConfig(pipeline_config);

    std::string downstream_host = pipeline_config.downstream_host;
    uint16_t downstream_port = pipeline_config.downstream_port;

    time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
    ProfileEvent::PushEvent(fmt::format("Send_buffer_id_{}_shape_{}_{}", hidden_unit->schedule_id,
                                        hidden_unit_host->shape_dims[0], hidden_unit_host->shape_dims[1]));
    Status status = client_raw_socket_->Send({downstream_host, downstream_port}, packet);
    ProfileEvent::PopEvent();
    time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
    KLLM_LOG_DEBUG << "DataChannel::ProcessSendPacketLoop send packet schedule_id:" << hidden_unit->schedule_id
                   << " cost time: " << end_time_ms - start_time_ms << " ms, shape:" << hidden_unit_host->shape_dims[0]
                   << ", " << hidden_unit_host->shape_dims[1];

    if (!status.OK()) {
      KLLM_LOG_ERROR << "DataChannel process send packet loop error, send packet failed. schedule_id="
                     << hidden_unit->schedule_id << ", info:" << status.GetMessage();
    }
    KLLM_LOG_DEBUG << "DataChannel::ProcessSendPacketLoop send success. schedule_id=" << hidden_unit->schedule_id;

    // Resue the packet buffer
    hidden_unit_buffer_pool_->FreeHostBuffer(packet);

    // Notify that send operation finished.
    hidden_unit_buffer_pool_->NotifySendFinished();
  }

  return Status();
}

}  // namespace ksana_llm
