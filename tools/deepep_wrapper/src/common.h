/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/types.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace deepep_wrapper {

#define EP_HOST_ASSERT(condition) assert(condition)
const size_t kMaxNumRanks = 256;
const size_t kNvshmemUniqudIdSize = 128;
const size_t kIpcHandlesSize = 64;
const size_t kNumMaxNvlPeers = 8;
// Adapted from DeepEP [https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py#L30]
const size_t kDeepEPDefaultNumSMs = 20;

#define CUDA_CHECK(call)                                                                                    \
  do {                                                                                                      \
    cudaError_t error = call;                                                                               \
    if (error != cudaSuccess) {                                                                             \
      std::ostringstream oss;                                                                               \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " in function '" << __FUNCTION__ << "()'\n" \
          << "Error code: " << error << " (" << cudaGetErrorName(error) << ")\n"                            \
          << "Error description: " << cudaGetErrorString(error);                                            \
      throw std::runtime_error(oss.str());                                                                  \
    }                                                                                                       \
  } while (0)

// 共享内存名称
#define SHM_NAME "/nvshmem_ipc_data"

// IPC数据结构
struct IPCData {
  pid_t child_pids[kMaxNumRanks];  // 全部子进程 pid

  // NVSHMEM 与 IPCHandles: 用于 DeepEP 组网
  uint8_t shared_unique_id[kMaxNumRanks][kNvshmemUniqudIdSize];
  volatile bool unique_id_ready[kMaxNumRanks];
  char ipc_handles[kMaxNumRanks][kIpcHandlesSize];
  volatile bool ipc_handle_ready[kMaxNumRanks];

  // 共享内存创建标记
  volatile bool ready = false;
  volatile bool process_finished[kMaxNumRanks];

  // 与一念做指令同步
  volatile bool trigger_init_hidden_buffers[kMaxNumRanks];
  volatile bool trigger_init_moe_buffer[kMaxNumRanks];
  volatile bool trigger_dispatch[kMaxNumRanks];
  volatile bool trigger_combine[kMaxNumRanks];
  volatile bool trigger_exit = false;

  // 进程间 event 同步信号
  cudaIpcEventHandle_t dispatch_events[kMaxNumRanks];
  cudaIpcEventHandle_t combine_events[kMaxNumRanks];

  // 错误状态
  volatile int error_code;
  char error_message[256];

  // 基础配置
  size_t num_experts;
  size_t num_topk;
  size_t hidden_size;

  // 用于 KsanaLLM 与 DeepEPWrapper 进程间交互的显存数据
  bool use_scales = false;
  cudaIpcMemHandle_t x[kMaxNumRanks];
  cudaIpcMemHandle_t x_scales[kMaxNumRanks];
  cudaIpcMemHandle_t topk_ids[kMaxNumRanks];
  cudaIpcMemHandle_t topk_weights[kMaxNumRanks];

  cudaIpcMemHandle_t hidden_buffer[kMaxNumRanks][3];
  volatile int input_buffer_idx[kMaxNumRanks];
  volatile int output_buffer_idx[kMaxNumRanks];
  cudaIpcMemHandle_t moe_buffer[kMaxNumRanks];

  volatile int recv_token_num[kMaxNumRanks];
};

// 公共函数声明
bool InitCuda(int rank);
void GetDeviceInfo(const std::string& info);

// 共享内存管理
IPCData* CreateSharedMemory(int node_rank);
IPCData* OpenSharedMemory(int node_rank);
void CleanupSharedMemory(int node_rank);

// 进程主函数声明
int Process(IPCData* shared_data, int num_ranks, int rank, int world_size, int node_rank);

}  // namespace deepep_wrapper
