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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// is_start: whether this is the very first synchronization barrier.
// need_fence: whether a memory fence is needed. If true, a release-acquire
// semantic is used to enforce memory access order before and after this
// barrier.
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank, uint32_t root_rank) {
  if constexpr (!is_start) __syncthreads();
  static_assert(!(is_start && need_fence));  // 开始屏障不应该需要内存屏障。
  if ((threadIdx.x >= root_rank) && (threadIdx.x - root_rank < ngpus)) {
    // Increment the counter. Technically we only need one counter, but we use
    // multiple per block to eliminate the need to share the counter via smem.
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);
      while (ld_flag_acquire(self_counter_ptr) != val)
        ;
    } else {
      st_flag_volatile(peer_counter_ptr, val);
      while (ld_flag_volatile(self_counter_ptr) != val)
        ;
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx, uint32_t root_rank) {
  A tmp = upcast(ptrs[root_rank][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i + root_rank][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank,
                               uint32_t root_rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank, root_rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx, root_rank);
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank, root_rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank,
                               uint32_t root_rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = (rank - root_rank) * part;
  int end = (rank - root_rank) == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = ((rank - root_rank) + i) % ngpus + root_rank;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank, root_rank);
  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx, 0);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank, root_rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from all
  // ranks.
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = (((rank - root_rank) + i) % ngpus + root_rank);
      if (gather_from_rank == (ngpus - 1 + root_rank) || idx < part) {
        int dst_idx = (gather_from_rank - root_rank) * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// Signals are an array of ipc-enabled buffers from all ranks.
// For each of the buffer, the layout is as follows:
// | -- sizeof(Signal) -- | ------ a few MB ----- |
// The first section is for allreduce synchronization, and the second section
// is for storing the intermediate results required by some allreduce algos.

// Note: this class does not own any device memory. Any required buffers
// are passed in from the constructor.
CustomAllreduce::CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size,
                                 bool full_nvlink, uint32_t root_rank)
    : rank_(rank),
      world_size_(world_size),
      full_nvlink_(full_nvlink),
      self_sg_(signals[rank]),
      root_rank_(root_rank),
      d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
      d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
  // NOTE(karlluo): in group all reduce, the world size is the group size.
  for (int i = 0; i < world_size_; i++) {
    sg_.signals[i + root_rank_] = signals[i + root_rank_];
  }
}

void CustomAllreduce::CheckRankDataCapacity(size_t num) {
  if (d_rank_data_base_ + num > d_rank_data_end_)
    throw std::runtime_error("Rank data buffer is overflowed by " +
                             std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
}

void CustomAllreduce::RegisterBuffer(void** ptrs, cudaStream_t& stream) {
  CheckRankDataCapacity();
  RankData data;
  for (uint32_t i = root_rank_; i < (root_rank_ + world_size_); ++i) {
    data.ptrs[i] = ptrs[i];
  }
  auto d_data = d_rank_data_base_++;
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice, stream));
  buffers_[ptrs[rank_]] = d_data;
}

// Performs allreduce, assuming input has already been registered.

// Block and grid default configs are results after careful grid search. Using
// 36 blocks give the best or close to the best runtime on the devices I
// tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also only
// take a small amount of SMs. Not quite sure the underlying reason, but my
// guess is that too many SMs will cause contention on NVLink bus.
template <typename T>
void CustomAllreduce::AllReduce(cudaStream_t stream, T* input, T* output, int size, int threads, int block_limit) {
  auto d = packed_t<T>::P::size;
  if (size % d != 0)
    throw std::runtime_error(
        "custom allreduce currently requires input length to be multiple "
        "of " +
        std::to_string(d));
  if (block_limit > kMaxBlocks)
    throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks) + ". Got " +
                             std::to_string(block_limit));

  RankData* ptrs;
  cudaStreamCaptureStatus status;
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamIsCapturing(stream, &status));
  if (status == cudaStreamCaptureStatusActive) {
    ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
    graph_unreg_buffers_.push_back(input);
  } else {
    auto it = buffers_.find(input);
    if (it == buffers_.end())
      throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                               " is not registered!");
    ptrs = it->second;
  }

  size /= d;
  auto bytes = size * sizeof(typename packed_t<T>::P);
  int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name) \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, root_rank_, size);
  // TODO(hanzhi713): Threshold is different for A100 and H100.
  // Add per device threshold.
#define REDUCE_CASE(ngpus)                                                                        \
  case ngpus: {                                                                                   \
    if (world_size_ == 2) {                                                                       \
      KL(ngpus, cross_device_reduce_1stage);                                                      \
    } else if (full_nvlink_) {                                                                    \
      if ((world_size_ <= 4 && bytes < 512 * 1024) || (world_size_ <= 8 && bytes < 256 * 1024)) { \
        KL(ngpus, cross_device_reduce_1stage);                                                    \
      } else {                                                                                    \
        KL(ngpus, cross_device_reduce_2stage);                                                    \
      }                                                                                           \
    }                                                                                             \
    break;                                                                                        \
  }

  switch (world_size_) {
    REDUCE_CASE(2)
    REDUCE_CASE(4)
    REDUCE_CASE(6)
    REDUCE_CASE(8)
    default:
      throw std::runtime_error(
          "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
          "gpus = " +
          std::to_string(world_size_));
  }
#undef REDUCE_CASE
#undef KL
}

CustomAllreduce::~CustomAllreduce() {}

// To inspect PTX/SASS, copy paste this header file to compiler explorer and add
// a template instantiation:
template void CustomAllreduce::AllReduce<float>(cudaStream_t, float*, float*, int, int, int);
template void CustomAllreduce::AllReduce<half>(cudaStream_t, half*, half*, int, int, int);
template void CustomAllreduce::AllReduce<__nv_bfloat16>(cudaStream_t, __nv_bfloat16*, __nv_bfloat16*, int, int, int);

}  // namespace nvidia
}  // namespace llm_kernels
