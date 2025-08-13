/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace llm_kernels {
namespace nvidia {

constexpr int kMaxBlocks = 36;
constexpr int maxDeviceCount = 8;

// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][maxDeviceCount];
  // Two sets of peer counters are needed for two syncs. The reason is that
  // it's possible for peer GPU block to arrive at the second sync point while
  // the current GPU block haven't passed the first sync point. Thus, peer GPU
  // may write counter+1 while current GPU is busy waiting for counter. We use
  // alternating counter array to avoid this possibility.
  alignas(128) FlagType peer_counter[2][kMaxBlocks][maxDeviceCount];
};

struct __align__(16) RankData {
  const void *__restrict__ ptrs[maxDeviceCount];
};

struct __align__(16) RankSignals {
  Signal *signals[maxDeviceCount];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;
  uint32_t root_rank_;

  RankSignals sg_;
  std::unordered_map<void *, RankData *> buffers_;
  Signal *self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph capture
  // time. Therefore, during capture, we increment the rank data pointer and use
  // that as the argument to the kernel. The kernel arguments are stored in
  // graph_unreg_buffers_. The actual peer pointers will be filled in at the
  // memory pointed to by the pointers in graph_unreg_buffers_ when
  // the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  RankData *saved_d_rank_data_base_, *saved_d_rank_data_end_;
  std::vector<void *> graph_unreg_buffers_;

  // Signals are an array of ipc-enabled buffers from all ranks.
  // For each of the buffer, the layout is as follows:
  // | -- sizeof(Signal) -- | ------ a few MB ----- |
  // The first section is for allreduce synchronization, and the second section
  // is for storing the intermediate results required by some allreduce algos.
  //
  // Note: this class does not own any device memory. Any required buffersare passed in from the constructor.
  CustomAllreduce(void *rank_data, size_t rank_data_sz, int rank, int world_size, bool full_nvlink = true,
                  uint32_t root_rank = 0);

  void RegisterSignalBuffer(Signal **signals);

  void CheckRankDataCapacity(size_t num = 1);

  void RegisterBuffer(void **ptrs, cudaStream_t &stream);

  // This is the result after careful grid search. Using 36 blocks give the best
  // or close to the best runtime on the devices I tried: A100, A10, A30, T4,
  // V100. You'll notice that NCCL kernels also only take a small amount of SMs.
  // Not quite sure the underlying reason, but my guess is that too many SMs
  // will cause contention on NVLink bus.
  template <typename T>
  void AllReduce(cudaStream_t stream, T *input, T *output, int size, int threads = 512, int block_limit = 36);

  ~CustomAllreduce();
};

}  // namespace nvidia
}  // namespace llm_kernels
