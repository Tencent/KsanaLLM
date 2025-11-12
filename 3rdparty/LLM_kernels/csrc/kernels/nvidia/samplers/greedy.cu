/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "greedy.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__device__ void InvokeWrapArgMax(volatile T* s_max_values, volatile uint32_t* s_argmax) {
  if (static_cast<T>(s_max_values[threadIdx.x]) < static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_WARP_SIZE];
  }
  if (static_cast<T>(s_max_values[threadIdx.x]) <
      static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_HALF_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_HALF_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_HALF_WARP_SIZE];
  }
  if (static_cast<T>(s_max_values[threadIdx.x]) <
      static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_QUARTER_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_QUARTER_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_QUARTER_WARP_SIZE];
  }
  if (static_cast<T>(s_max_values[threadIdx.x]) <
      static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_EIGHTH_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_EIGHTH_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_ONE_EIGHTH_WARP_SIZE];
  }
  if (static_cast<T>(s_max_values[threadIdx.x]) <
      static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_SIXTEENTH_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_SIXTEENTH_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_ONE_SIXTEENTH_WARP_SIZE];
  }
  if (static_cast<T>(s_max_values[threadIdx.x]) <
      static_cast<T>(s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_THIRTY_TWO_WARP_SIZE])) {
    s_max_values[threadIdx.x] = s_max_values[threadIdx.x + DEFAULT_CUDA_ONE_THIRTY_TWO_WARP_SIZE];
    s_argmax[threadIdx.x] = s_argmax[threadIdx.x + DEFAULT_CUDA_ONE_THIRTY_TWO_WARP_SIZE];
  }
}

template <typename T>
__global__ void InvokeNewArgMaxReduceKernel(const T* input, const int32_t batch_size, const int32_t vocab_size,
                                            uint32_t* result) {
  // NOTE(karlluo): shm consist with DEFAULT_CUDA_BLOCK_THREADS_NUM (float + uint32_t) as following:
  // |-- blockDim.x float --|-- blockDim.x uin32_t --|
  // |     for max value    |    for max index     --|
  // prevent from bank conflict, each thread handle one element `for max value` and `for max index`
  extern __shared__ unsigned char smem[];
  T* s_max_values = reinterpret_cast<T*>(smem);
  uint32_t* s_argmax = reinterpret_cast<uint32_t*>(smem + sizeof(T) * blockDim.x);

  const uint32_t row = blockIdx.x;
  const T* __restrict__ d_value = input + row * vocab_size;
  uint32_t* __restrict__ d_index = &(result[row]);

  // Initialize per-thread best
  uint32_t max_id = 0u;
  T max_value = NegativeInfinity<T>();

  // Coalesced vocab chunk scan with software unrolling (4x)
  const uint32_t threads = blockDim.x;
  constexpr uint32_t kTileSize = 4;
  for (uint32_t vocab_chunk_offset = 0u; vocab_chunk_offset < static_cast<uint32_t>(vocab_size);
       vocab_chunk_offset += threads * kTileSize) {
    uint32_t idx_0 = vocab_chunk_offset + threadIdx.x;
    if (idx_0 < static_cast<uint32_t>(vocab_size)) {
      T v0 = d_value[idx_0];
      if (max_value < v0) {
        max_value = v0;
        max_id = idx_0;
      }
    }
    uint32_t idx_1 = idx_0 + threads;
    if (idx_1 < static_cast<uint32_t>(vocab_size)) {
      T v1 = d_value[idx_1];
      if (max_value < v1) {
        max_value = v1;
        max_id = idx_1;
      }
    }
    uint32_t idx_2 = idx_1 + threads;
    if (idx_2 < static_cast<uint32_t>(vocab_size)) {
      T v2 = d_value[idx_2];
      if (max_value < v2) {
        max_value = v2;
        max_id = idx_2;
      }
    }
    uint32_t idx_3 = idx_2 + threads;
    if (idx_3 < static_cast<uint32_t>(vocab_size)) {
      T v3 = d_value[idx_3];
      if (max_value < v3) {
        max_value = v3;
        max_id = idx_3;
      }
    }
  }

  s_max_values[threadIdx.x] = max_value;
  s_argmax[threadIdx.x] = max_id;

  __syncthreads();

  for (uint32_t border = blockDim.x >> 1; border > DEFAULT_CUDA_WARP_SIZE; border >>= 1) {
    if (threadIdx.x < border) {
      const uint32_t compare_idx = threadIdx.x + border;
      if (compare_idx < blockDim.x) {
        const T v = s_max_values[compare_idx];
        if (s_max_values[threadIdx.x] < v) {
          s_max_values[threadIdx.x] = v;
          s_argmax[threadIdx.x] = s_argmax[compare_idx];
        }
      }
    }
    __syncthreads();
  }

  // Final warp-level reduction using stable tie-break helper
  if (threadIdx.x < DEFAULT_CUDA_WARP_SIZE) {
    InvokeWrapArgMax(s_max_values, s_argmax);
  }

  if (threadIdx.x == 0) {
    *d_index = s_argmax[0];
  }
}

template <typename T>
__global__ void InvokeOldArgMaxReduceKernel(const T* input, const int32_t batch_size, const int32_t vocab_size,
                                            uint32_t* result) {
  if (threadIdx.x > vocab_size) {
    return;
  }

  uint32_t border = vocab_size >> 1;

  // NOTE(karlluo): shm consist with DEFAULT_CUDA_BLOCK_THREADS_NUM (float + uint32_t) as following:
  // |-- blockDim.x float --|-- blockDim.x uin32_t --|
  // |     for max value    |    for max index     --|
  // prevent from bank conflict, each thread handle one element `for max value` and `for max index`
  extern __shared__ uint32_t argmax_shm[];
  uint32_t* s_argmax = reinterpret_cast<uint32_t*>(&argmax_shm[blockDim.x]);
  T* s_max_values = reinterpret_cast<T*>(&argmax_shm[0]);

  // NOTE(karlluo): get real value pointer
  uint32_t pos = blockIdx.x;
  T* d_value = const_cast<T*>(input + pos * vocab_size);
  uint32_t* d_index = &(result[blockIdx.x]);

  // NOTE(karlluo): init idx
  uint32_t max_id = threadIdx.x;
  T max_value = d_value[threadIdx.x];

  // NOTE(karlluo): reduce all to shm
  for (uint32_t idx = threadIdx.x; idx < vocab_size; idx += blockDim.x) {
    if (idx < vocab_size && max_value < d_value[idx]) {
      max_id = idx;
      max_value = d_value[idx];
    }
  }

  s_max_values[threadIdx.x] = max_value;
  s_argmax[threadIdx.x] = max_id;

  // NOTE(karlluo): reduce all shm to 32 threads shm
  // get argmax with binary tree
  // each half thread compare the rest half data
  uint32_t compare_idx = max_id;
  for (border = blockDim.x >> 1; border > DEFAULT_CUDA_WARP_SIZE; border >>= 1) {
    if (threadIdx.x > border) {
      return;
    }
    compare_idx = border + threadIdx.x;
    __syncthreads();

    if (compare_idx < blockDim.x && max_value < s_max_values[compare_idx]) {
      max_value = s_max_values[compare_idx];
      max_id = s_argmax[compare_idx];
    }
    s_max_values[threadIdx.x] = max_value;
    s_argmax[threadIdx.x] = max_id;
  }

  // NOTE(karlluo): reduce shm[0, ..., 31] to shm[0]
  if (threadIdx.x < DEFAULT_CUDA_WARP_SIZE) {
    InvokeWrapArgMax(s_max_values, s_argmax);
  }

  if (threadIdx.x == 0) {
    *d_index = static_cast<uint64_t>(s_argmax[0]);
  }
}

template <typename T>
using ArgMaxPair = cub::KeyValuePair<int, T>;  // (idx, val)

template <typename T>
__global__ void InvokeArgMaxReduceKernel(const T* input, const int32_t batch_size, const int32_t vocab_size,
                                         uint32_t* result) {
  using BlockReduce = cub::BlockReduce<ArgMaxPair<T>, DEFAULT_CUDA_BLOCK_THREADS_NUM>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // First reduce in each thread.
  const int offset = blockIdx.x * vocab_size;
  int idx = 0;
  T val = input[offset];
  for (int compare_idx = threadIdx.x; compare_idx < vocab_size; compare_idx += blockDim.x) {
    T compare_val = input[offset + compare_idx];
    if (val < compare_val) {
      idx = compare_idx;
      val = compare_val;
    }
  }

  // Then reduce in the block.
  cub::ArgMax argmax_op;
  idx = BlockReduce(temp_storage).Reduce(ArgMaxPair<T>{idx, val}, argmax_op).key;

  // Write result to global memory.
  if (threadIdx.x == 0) {
    result[blockIdx.x] = idx;
  }
}

template <typename T>
void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result,
                        cudaStream_t& stream) {
  dim3 grid(batch_size);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);

  // By default, the old version of argmax is used, which has issues with multiple maxima.
  // We will transition to the correct new version later.
  if (std::getenv("USE_OLD_ARGMAX") != nullptr) {
    const uint32_t s_mem_size = DEFAULT_CUDA_BLOCK_THREADS_NUM * (sizeof(float) + sizeof(uint32_t));
    InvokeOldArgMaxReduceKernel<<<grid, block, s_mem_size, stream>>>(input, batch_size, vocab_size, result);
  } else {
    const uint32_t s_mem_size = block.x * (sizeof(T) + sizeof(uint32_t));
    InvokeNewArgMaxReduceKernel<<<grid, block, s_mem_size, stream>>>(input, batch_size, vocab_size, result);
  }
}

#define INSTANTIATE_INVOKE_ARG_MAX_REDUCE(T)                                                           \
  template void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, \
                                   uint32_t* result, cudaStream_t& stream);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(float);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(half);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(__nv_bfloat16);
#undef INSTANTIATE_INVOKE_ARG_MAX_REDUCE

template <typename T>
__global__ void LocalArgMaxReduceKernel(const T* __restrict__ input, const int32_t vocab_size,
                                        const int32_t vocab_size_pad, const int32_t rank, float* __restrict__ result) {
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t thread_num = blockDim.x;

  T max_val = NegativeInfinity<T>();
  uint32_t max_idx = 0;
  // Argmax in each thread
  // Use this style instead of vectorization to maintain precision
  const T* d_value = input + batch_idx * vocab_size;
  constexpr uint32_t kTileSize = 4;
  for (uint32_t vocab_idx = threadIdx.x; vocab_idx < vocab_size; vocab_idx += thread_num * kTileSize) {
    const uint32_t compare_idx0 = vocab_idx;
    if (compare_idx0 < vocab_size) {
      const T compare_val0 = d_value[compare_idx0];
      if (max_val < compare_val0) {
        max_val = compare_val0;
        max_idx = compare_idx0;
      }
    }
    const uint32_t compare_idx1 = compare_idx0 + thread_num;
    if (compare_idx1 < vocab_size) {
      const T compare_val1 = d_value[compare_idx1];
      if (max_val < compare_val1) {
        max_val = compare_val1;
        max_idx = compare_idx1;
      }
    }
    const uint32_t compare_idx2 = compare_idx1 + thread_num;
    if (compare_idx2 < vocab_size) {
      const T compare_val2 = d_value[compare_idx2];
      if (max_val < compare_val2) {
        max_val = compare_val2;
        max_idx = compare_idx2;
      }
    }
    const uint32_t compare_idx3 = compare_idx2 + thread_num;
    if (compare_idx3 < vocab_size) {
      const T compare_val3 = d_value[compare_idx3];
      if (max_val < compare_val3) {
        max_val = compare_val3;
        max_idx = compare_idx3;
      }
    }
  }
  extern __shared__ uint8_t smem[];
  T* s_max_values = reinterpret_cast<T*>(smem);
  uint32_t* s_argmax = reinterpret_cast<uint32_t*>(smem + sizeof(T) * thread_num);
  s_max_values[threadIdx.x] = max_val;
  s_argmax[threadIdx.x] = max_idx;

  // Argmax across threads
  for (uint32_t border = thread_num >> 1; border > DEFAULT_CUDA_WARP_SIZE; border >>= 1) {
    if (threadIdx.x >= border) {
      return;
    }
    const uint32_t compare_idx = border + threadIdx.x;
    __syncthreads();

    const T compare_val = s_max_values[compare_idx];
    if (max_val < compare_val) {
      max_val = compare_val;
      max_idx = s_argmax[compare_idx];
    }
    s_max_values[threadIdx.x] = max_val;
    s_argmax[threadIdx.x] = max_idx;
  }

  // Argmax in warp
  if (threadIdx.x < DEFAULT_CUDA_WARP_SIZE) {
    InvokeWrapArgMax(s_max_values, s_argmax);
  }

  // Write the local max and idx
  if (threadIdx.x == 0) {
    result[batch_idx * 2] = static_cast<float>(s_max_values[0]);
    result[batch_idx * 2 + 1] = static_cast<float>(rank * vocab_size_pad + s_argmax[0]);
  }
}

template <typename T>
void InvokeLocalArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size,
                             const int32_t vocab_size_pad, const int32_t rank, float* result, cudaStream_t stream) {
  dim3 grid(batch_size);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  const uint32_t s_mem_size = DEFAULT_CUDA_BLOCK_THREADS_NUM * (sizeof(T) + sizeof(uint32_t));
  LocalArgMaxReduceKernel<<<grid, block, s_mem_size, stream>>>(input, vocab_size, vocab_size_pad, rank, result);
}
#define INSTANTIATE_INVOKE_LOCAL_ARG_MAX_REDUCE(T)                                                                    \
  template void InvokeLocalArgMaxReduce(const T*, const int32_t, const int32_t, const int32_t, const int32_t, float*, \
                                        cudaStream_t)
INSTANTIATE_INVOKE_LOCAL_ARG_MAX_REDUCE(float);
INSTANTIATE_INVOKE_LOCAL_ARG_MAX_REDUCE(half);
INSTANTIATE_INVOKE_LOCAL_ARG_MAX_REDUCE(__nv_bfloat16);
#undef INSTANTIATE_INVOKE_ARG_MAX_REDUCE

template <int32_t TP_SIZE>
__global__ void WarpArgMaxReduceKernel(const float* __restrict__ input, uint32_t* __restrict__ result) {
  const uint32_t token_idx = blockIdx.x;
  const uint32_t lane_idx = threadIdx.x;

  const uint32_t offset = 2 * (token_idx * TP_SIZE + lane_idx);
  float max_val = input[offset];
  float max_idx = input[offset + 1];
  // Compare in warp
  constexpr uint32_t mask = 0xffff;
  if constexpr (TP_SIZE >= 16) {
    const float other_val = __shfl_down_sync(mask, max_val, 8);
    const float other_idx = __shfl_down_sync(mask, max_idx, 8);
    if (max_val < other_val) {
      max_val = other_val;
      max_idx = other_idx;
    }
  }
  if constexpr (TP_SIZE >= 8) {
    const float other_val = __shfl_down_sync(mask, max_val, 4);
    const float other_idx = __shfl_down_sync(mask, max_idx, 4);
    if (max_val < other_val) {
      max_val = other_val;
      max_idx = other_idx;
    }
  }
  if constexpr (TP_SIZE >= 4) {
    const float other_val = __shfl_down_sync(mask, max_val, 2);
    const float other_idx = __shfl_down_sync(mask, max_idx, 2);
    if (max_val < other_val) {
      max_val = other_val;
      max_idx = other_idx;
    }
  }
  if constexpr (TP_SIZE >= 2) {
    const float other_val = __shfl_down_sync(mask, max_val, 1);
    const float other_idx = __shfl_down_sync(mask, max_idx, 1);
    if (max_val < other_val) {
      max_val = other_val;
      max_idx = other_idx;
    }
  }

  // Write the final argmax
  if (lane_idx == 0) {
    result[token_idx] = static_cast<uint32_t>(max_idx);
  }
}

void InvokeWarpArgMaxReduce(const float* input, const int32_t batch_size, const int32_t tp_size, uint32_t* result,
                            cudaStream_t stream) {
  dim3 grid(batch_size);
  switch (tp_size) {
    case 2:
      WarpArgMaxReduceKernel<2><<<grid, dim3(2), 0, stream>>>(input, result);
      break;
    case 4:
      WarpArgMaxReduceKernel<4><<<grid, dim3(4), 0, stream>>>(input, result);
      break;
    case 6:
      WarpArgMaxReduceKernel<6><<<grid, dim3(6), 0, stream>>>(input, result);
      break;
    case 8:
      WarpArgMaxReduceKernel<8><<<grid, dim3(8), 0, stream>>>(input, result);
      break;
    case 16:
      WarpArgMaxReduceKernel<16><<<grid, dim3(16), 0, stream>>>(input, result);
      break;
    default:
      KLLM_KERNEL_THROW("Unsupported tp size: %d", tp_size);
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
