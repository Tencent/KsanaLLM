/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/packet_util.h"
#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif

#include <cstdlib>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

Packet* GetPacketObject(PacketType packet_type, size_t body_size) {
  Packet* packet = nullptr;
  switch (packet_type) {
    case PacketType::DATA_REQ_HIDDEN_UNIT: {
      packet = GetHiddenUnitBufferPool()->GetHostBuffer();
      if (packet != nullptr) {
        packet->size = GetHiddenUnitBufferPool()->GetHostPacketSize(packet);
      }
      break;
    }
    case PacketType::PACKET_TYPE_UNKNOWN:
    case PacketType::CONTROL_REQ_SCHEDULE: {
      packet = GetRawPacket(body_size);
      if (packet != nullptr) {
        packet->size = body_size;
      }
      break;
    }
    case PacketType::CONTROL_REQ_BARRIER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(BarrierRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(BarrierRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_BARRIER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(BarrierResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(BarrierResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_BLOCKNUM: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(CacheBlockNumRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(CacheBlockNumRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_BLOCKNUM: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(CacheBlockNumResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(CacheBlockNumResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_LAYER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(AllocateLayerRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(AllocateLayerRequest);
      }
      break;
    }
    case PacketType::CONTROL_REQ_ADD_NODE: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(AddNodeRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(AddNodeRequest);
      }
      break;
    }
    case PacketType::CONTROL_REQ_HEARTBEAT: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(HeartbeatRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(HeartbeatRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_HEARTBEAT: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(HeartbeatResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(HeartbeatResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_DEEPEP_META: {
      size_t nvshmem_unique_id_size = kNvshmemUniqudIdSize;
      size_t ipc_handles_size = kMaxNumRanks * kIpcHandlesSize;
      packet = reinterpret_cast<Packet*>(
          malloc(sizeof(Packet) + nvshmem_unique_id_size + ipc_handles_size + sizeof(NvshmemUniqueIdRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(NvshmemUniqueIdRequest) + nvshmem_unique_id_size + ipc_handles_size;
      }
      break;
    }
    case PacketType::CONTROL_RSP_DEEPEP_META: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(NvshmemUniqueIdResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(NvshmemUniqueIdResponse);
      }
      break;
    }
    case PacketType::DATA_RSP_HIDDEN_UNIT:
    case PacketType::CONTROL_RSP_SCHEDULE:
    case PacketType::CONTROL_RSP_LAYER:
    case PacketType::CONTROL_RSP_ADD_NODE:
    case PacketType::CONTROL_REQ_SHUTDOWN:
    case PacketType::CONTROL_RSP_SHUTDOWN: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet)));
      if (packet != nullptr) {
        packet->size = 0;
      }
      break;
    }
    default:
      return nullptr;
  }

  if (packet != nullptr) {
    packet->type = packet_type;
  }
  return packet;
}

}  // namespace ksana_llm
