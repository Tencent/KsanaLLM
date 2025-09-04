/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// 用于与 deepep_wrapper 通信的共享内存结构体
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
#ifdef ENABLE_CUDA
  cudaIpcEventHandle_t dispatch_events[kMaxNumRanks];
  cudaIpcEventHandle_t combine_events[kMaxNumRanks];
#endif

  // 错误状态
  volatile int error_code;
  char error_message[256];

  // 基础配置
  size_t num_experts;
  size_t num_topk;
  size_t hidden_size;

  // 用于 KsanaLLM 与 DeepEPWrapper 进程间交互的显存数据
  bool use_scales = false;
#ifdef ENABLE_CUDA
  cudaIpcMemHandle_t x[kMaxNumRanks];
  cudaIpcMemHandle_t x_scales[kMaxNumRanks];
  cudaIpcMemHandle_t topk_ids[kMaxNumRanks];
  cudaIpcMemHandle_t topk_weights[kMaxNumRanks];

  cudaIpcMemHandle_t hidden_buffer[kMaxNumRanks][3];
  volatile int input_buffer_idx[kMaxNumRanks];
  volatile int output_buffer_idx[kMaxNumRanks];
  cudaIpcMemHandle_t moe_buffer[kMaxNumRanks];
#endif

  volatile int recv_token_num[kMaxNumRanks];
};

class ExpertParallelDeepepWrapper {
 public:
  ExpertParallelDeepepWrapper(size_t num_ranks, size_t num_ranks_per_node, size_t node_rank, size_t max_token_num,
                              size_t hidden_size, size_t expert_topk, size_t num_experts,
                              std::shared_ptr<Context> context);

  ~ExpertParallelDeepepWrapper();

  void Init();

  virtual Status SetReady();

  Status SetHiddenBuffers(const std::vector<Tensor>& hidden_buffers, int rank);

  Status SetMoeBuffer(const std::vector<Tensor>& moe_buffers, int rank);

  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, int rank);

  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors, int rank);

  virtual uint8_t* GetNvshmemUniqueId() { return reinterpret_cast<uint8_t*>(shared_data_->shared_unique_id); }

  virtual char* GetIPCHandles() { return reinterpret_cast<char*>(shared_data_->ipc_handles); }

 private:
  std::string shm_name_;
  IPCData* shared_data_;
  size_t hidden_size_;
  size_t expert_topk_;
  size_t num_ranks_;
  size_t num_ranks_per_node_;
  size_t node_rank_;
  size_t max_token_num_;
  size_t num_experts_;
  bool initialized_;

  std::vector<void*> x_scales_ptrs_;
  std::vector<void*> topk_ids_ptrs_;
  std::vector<void*> topk_weights_ptrs_;
  std::shared_ptr<Context> context_;
  std::vector<std::unordered_map<void*, int>> tensor_address_to_id_;

#ifdef ENABLE_CUDA
  // 进程间 event 同步信号
  std::vector<cudaEvent_t> dispatch_events_;
  std::vector<cudaEvent_t> combine_events_;
#endif
};

}  // namespace ksana_llm
