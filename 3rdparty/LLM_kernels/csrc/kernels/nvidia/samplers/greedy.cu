/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "greedy.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>
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
  if (std::getenv("ENABLE_NEW_ARGMAX") == nullptr) {
    const uint32_t s_mem_size = DEFAULT_CUDA_BLOCK_THREADS_NUM * (sizeof(float) + sizeof(uint32_t));
    InvokeOldArgMaxReduceKernel<<<grid, block, s_mem_size, stream>>>(input, batch_size, vocab_size, result);
  } else {
    InvokeArgMaxReduceKernel<<<grid, block, 0, stream>>>(input, batch_size, vocab_size, result);
  }
}

#define INSTANTIATE_INVOKE_ARG_MAX_REDUCE(T)                                                           \
  template void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, \
                                   uint32_t* result, cudaStream_t& stream);

INSTANTIATE_INVOKE_ARG_MAX_REDUCE(float);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(half);
INSTANTIATE_INVOKE_ARG_MAX_REDUCE(__nv_bfloat16);

#undef INSTANTIATE_INVOKE_ARG_MAX_REDUCE

}  // namespace nvidia
}  // namespace llm_kernels
