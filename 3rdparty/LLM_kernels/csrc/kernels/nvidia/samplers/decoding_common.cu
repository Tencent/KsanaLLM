/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/decodingCommon.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/common/reduceKernelUtils.cuh
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <cub/cub.cuh>

#include "decoding_common.h"

namespace tensorrt_llm {
namespace kernels {
static float constexpr HALF_FLT_MAX = 65504.F;
static int32_t constexpr BLOCK_SIZE = 256;
#define FINAL_MASK 0xffffffff

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template <typename T>
__inline__ __device__ T WarpReduceSumKernel(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));  //__shfl_sync bf16 return float when sm < 80
  return val;
}

// Calculate the sum of all elements in a block
template <typename T>
__inline__ __device__ T BlockReduceSumKernel(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = WarpReduceSumKernel<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = WarpReduceSumKernel<T>(val);

  return val;
}

template <typename T>
__inline__ __device__ T WarpReduceMaxKernel(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

// Calculate the maximum of all elements in a block
template <typename T>
__inline__ __device__ T BlockReduceMaxKernel(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = WarpReduceMaxKernel(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = WarpReduceMaxKernel(val);

  return val;
}

__global__ void InitCurandKernel(curandState_t* state, const int* batch_slots, const int size,
                                 const uint64_t randomSeed) {
  int const idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    auto const batch_slot = batch_slots != nullptr ? batch_slots[idx] : idx;
    curand_init(randomSeed, 0, 0, &state[batch_slot]);
  }
}

void InvokeCurandInitialize(curandState_t* state, const int* batch_slots, const size_t batch_size,
                            const uint64_t randomSeed, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((int)(ceil(batch_size * 1.0 / BLOCK_SIZE)));
  InitCurandKernel<<<grid, block, 0, stream>>>(state, batch_slots, batch_size, randomSeed);
}

__global__ void InitCurandBatch(curandState_t* states, const int* batch_slots, const int size,
                                const uint64_t* random_seeds) {
  int const idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    auto const batch_slot = batch_slots != nullptr ? batch_slots[idx] : idx;
    curand_init(random_seeds[batch_slot], 0, 0, &states[batch_slot]);
  }
}

void InvokeCurandBatchInitialize(curandState_t* states, const int* batch_slots, const size_t batch_size,
                                 const uint64_t* random_seeds, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid((int)(ceil(batch_size * 1.0 / BLOCK_SIZE)));
  InitCurandBatch<<<grid, block, 0, stream>>>(states, batch_slots, batch_size, random_seeds);
}

template <typename T>
__global__ void AddBiasSoftmaxKernel(T* logits, T** logits_ptrs, T* temperatures, T const* bias, int32_t const* end_ids,
                                     FinishedState const* finished, int32_t const* batch_slots, int32_t batch_size,
                                     int32_t max_batch_size, int32_t beam_width, int32_t vocab_size,
                                     int32_t vocab_size_padded, bool skip_soft_max, bool batch_slots_logits) {
  auto const batch_idx = blockIdx.x;
  auto const beam_idx = blockIdx.y;
  auto const batch_slot = batch_slots != nullptr ? batch_slots[batch_idx] : batch_idx;
  auto const batch_idx_logits = batch_slots_logits ? batch_slot : batch_idx;
  FinishedState const finish_state =
      finished != nullptr ? finished[beam_idx * max_batch_size + batch_slot] : FinishedState::empty();
  if (finish_state.isSkipDecoding()) {
    return;
  }

  auto logits_ptr = logits_ptrs ? logits_ptrs[batch_idx] + beam_idx * vocab_size_padded
                                : logits + (batch_idx_logits * beam_width + beam_idx) * vocab_size_padded;

  T temperature = temperatures ? temperatures[batch_idx] : T(1.0f);
  temperature = temperature == T(0.0f) ? T(1.0f) : temperature;
  bool finish = finish_state.isFinished();

  float max_val = -1 * FLT_MAX;
  bool const IS_FP16 = std::is_same<T, half>::value;
  T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  __shared__ float s_max_val;
  __shared__ float s_sum_val;

  for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
    logits_ptr[tid] = logits_ptr[tid] / temperature;
    auto logit = logits_ptr[tid];
    if (tid < vocab_size) {
      if (finish && end_ids != nullptr) {
        logit = (tid == end_ids[batch_slot]) ? MAX_T_VAL : -MAX_T_VAL;
      } else {
        T bias_val = (bias != nullptr) ? bias[tid] : (T)0.0f;
        logit += bias_val;
      }
    } else {
      logit = -MAX_T_VAL;
    }
    max_val = max(max_val, (float)logit);
    logits_ptr[tid] = logit;
  }

  if (!skip_soft_max) {
    max_val = BlockReduceMaxKernel<float>((float)max_val);
    if (threadIdx.x == 0) {
      s_max_val = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;
    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
      logits_ptr[tid] = __expf((float)logits_ptr[tid] - s_max_val);
      sum_val += (float)logits_ptr[tid];
    }

    sum_val = BlockReduceSumKernel<float>(sum_val);
    if (threadIdx.x == 0) {
      s_sum_val = sum_val;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
      logits_ptr[tid] = ((float)logits_ptr[tid] / (s_sum_val + 1e-6f));
    }
  }
}

template <typename T>
void InvokeAddBiasSoftMax(T* logits, T** logits_ptrs, T* temperatures, T const* bias, int32_t const* end_ids,
                          FinishedState const* finished, int32_t const* batch_slots, int32_t batch_size,
                          int32_t max_batch_size, int32_t beam_width, int32_t vocab_size, int32_t vocab_size_padded,
                          bool skip_soft_max, bool batch_slots_logits, cudaStream_t stream) {
  dim3 grid(batch_size, beam_width);
  auto const vocab_rounded_to_warp = (vocab_size + 31) & ~31;
  dim3 block(min(vocab_rounded_to_warp, 1024));
  // vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
  AddBiasSoftmaxKernel<<<grid, block, 0, stream>>>(logits, logits_ptrs, temperatures, bias, end_ids, finished,
                                                   batch_slots, batch_size, max_batch_size, beam_width, vocab_size,
                                                   vocab_size_padded, skip_soft_max, batch_slots_logits);
}

template void InvokeAddBiasSoftMax(float* logits, float** logits_ptrs, float* temperatures, float const* bias,
                                   int32_t const* end_ids, FinishedState const* finished, int32_t const* batch_slots,
                                   int32_t batch_size, int32_t max_batch_size, int32_t beam_width, int32_t vocab_size,
                                   int32_t vocab_size_padded, bool skip_soft_max, bool batch_slots_logits,
                                   cudaStream_t stream);

template void InvokeAddBiasSoftMax(half* logits, half** logits_ptrs, half* temperatures, half const* bias,
                                   int32_t const* end_ids, FinishedState const* finished, int32_t const* batch_slots,
                                   int32_t batch_size, int32_t max_batch_size, int32_t beam_width, int32_t vocab_size,
                                   int32_t vocab_size_padded, bool skip_soft_max, bool batch_slots_logits,
                                   cudaStream_t stream);

}  // namespace kernels
}  // namespace tensorrt_llm
