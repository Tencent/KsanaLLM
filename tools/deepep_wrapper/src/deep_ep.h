/*
 * Copyright (c) 2025 DeepSeek
 *
 * MIT License
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Adapted from
 * [DeepEP Project] https://github.com/deepseek-ai/DeepEP/blob/main/csrc/deep_ep.hpp
 */
#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include <torch/types.h>
#include <tuple>
#include <vector>
#include "common.h"

#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace deepep_wrapper {

struct Buffer {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

 public:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;

  // NVLink Buffer
  int64_t num_nvl_bytes;
  void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;

  // NVSHMEM Buffer
  int64_t num_rdma_bytes;
  void* rdma_buffer_ptr = nullptr;

  // Device info and communication
  int device_id;
  int num_device_sms;
  int rank, rdma_rank, nvl_rank;
  int num_ranks, num_rdma_ranks, num_nvl_ranks;
  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

  // Stream for communication
  at::cuda::CUDAStream comm_stream;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Barrier signals
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** barrier_signal_ptrs_gpu = nullptr;

  // Workspace
  void* workspace = nullptr;

  // Host-side MoE info
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped = nullptr;

  // Host-side expert-level MoE info
  volatile int* moe_recv_expert_counter = nullptr;
  int* moe_recv_expert_counter_mapped = nullptr;

  // Host-side RDMA-level MoE info
  volatile int* moe_recv_rdma_counter = nullptr;
  int* moe_recv_rdma_counter_mapped = nullptr;

 public:
  Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
         at::cuda::CUDAStream comm_stream);

  ~Buffer() noexcept(false);

  bool is_available() const;

  bool is_internode_available() const;

  int get_num_rdma_ranks() const;

  int get_rdma_rank() const;

  int get_root_rdma_rank(bool global) const;

  int get_local_device_id() const;

  std::string get_local_ipc_handle() const;

  torch::Stream get_comm_stream() const;

  void sync(IPCData* shared_data);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor,
             std::optional<deep_ep::EventHandle>>
  get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                      std::optional<deep_ep::EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

  std::tuple<int, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<deep_ep::EventHandle>>
  intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                     const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                     const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank,
                     const std::optional<torch::Tensor>& num_tokens_per_expert, int cached_num_recv_tokens,
                     const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_channel_prefix_matrix, int expert_alignment,
                     int num_worst_tokens, const deep_ep::Config& config,
                     std::optional<deep_ep::EventHandle>& previous_event, bool async, bool allocate_on_comm_stream,
                     void* output);

  std::tuple<int, std::optional<torch::Tensor>, std::optional<deep_ep::EventHandle>> intranode_combine(
      const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
      const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
      const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
      const torch::Tensor& send_head, const deep_ep::Config& config,
      std::optional<deep_ep::EventHandle>& previous_event, bool async, bool allocate_on_comm_stream, void* output);

  std::tuple<int, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
             std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<deep_ep::EventHandle>>
  internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                     const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                     const std::optional<torch::Tensor>& num_tokens_per_rank,
                     const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                     const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                     int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                     const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                     const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum, int expert_alignment,
                     const deep_ep::Config& config, std::optional<deep_ep::EventHandle>& previous_event, bool async,
                     bool allocate_on_comm_stream, void* output);

  std::tuple<int, std::optional<torch::Tensor>, std::optional<deep_ep::EventHandle>> internode_combine(
      const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
      const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
      const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
      const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum,
      const torch::Tensor& gbl_channel_prefix_matrix, const torch::Tensor& combined_rdma_head,
      const torch::Tensor& combined_nvl_head, const deep_ep::Config& config,
      std::optional<deep_ep::EventHandle>& previous_event, bool async, bool allocate_on_comm_stream, void* output);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<deep_ep::EventHandle>, std::optional<std::function<void()>>>
  low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                       const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                       int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8, bool round_scale,
                       bool use_ue8m0, bool async, bool return_recv_hook);

  std::tuple<torch::Tensor, std::optional<deep_ep::EventHandle>, std::optional<std::function<void()>>>
  low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                      const torch::Tensor& src_info, const torch::Tensor& layout_range,
                      int num_max_dispatch_tokens_per_rank, int num_experts, bool zero_copy, bool async,
                      bool return_recv_hook, const std::optional<torch::Tensor>& out = std::nullopt);

  torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden,
                                                    int num_experts) const;
};

}  // namespace deepep_wrapper
