/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stddef.h>

#include <cstdint>

#include "ksana_llm/distributed/packet_type.h"

namespace ksana_llm {

constexpr int kMaxNumRanks = 256;  // Maximum number of ranks for DeepEP networking
constexpr int kNcclUniqueIdSize = 128;
constexpr int kNvshmemUniqudIdSize = 128;
constexpr int kIpcHandlesSize = 64;

// For barrier
struct BarrierRequest {
  int node_rank;
  int clock_idx;
};

struct BarrierResponse {
  int clock_idx;
};

// for layer allocation, master to worker.
struct AllocateLayerRequest {
  uint16_t lower_layer_idx;
  uint16_t upper_layer_idx;

  int16_t lower_nextn_layer_idx;
  int16_t upper_nextn_layer_idx;

  char downstream_host[kMaxNumRanks];
  uint16_t downstream_port;

  // Used to broadcast nccl unique_id to all workers.
  // Because ncclUniqueId is platform dependency, so char[] is used.
  char nccl_unique_id[kNcclUniqueIdSize];
};

// add node, worker to master.
struct AddNodeRequest {
  std::size_t node_rank;

  char data_host[kMaxNumRanks];
  uint16_t data_port;
};

// del node
struct DelNodeRequest {
  size_t node_rank;
};

// make sure cache block num.
struct CacheBlockNumRequest {
  size_t node_rank;
  size_t device_block_num;
  size_t host_block_num;
};

// cache block num.
struct CacheBlockNumResponse {
  size_t device_block_num;
  size_t host_block_num;
};

// heartbeat
struct HeartbeatRequest {
  size_t node_rank;
};

// Same as req
struct HeartbeatResponse {
  size_t node_rank;
};

struct NvshmemUniqueIdRequest {
  int node_rank;
  char ipc_handles[kMaxNumRanks][kIpcHandlesSize];
  uint8_t nvshmem_unique_id[kMaxNumRanks][kNvshmemUniqudIdSize];
};

struct NvshmemUniqueIdResponse {
  int status;
};

}  // namespace ksana_llm
