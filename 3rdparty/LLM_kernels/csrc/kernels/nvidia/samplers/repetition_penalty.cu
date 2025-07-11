/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "repetition_penalty.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void RepetitionPenaltyKernel(const T* logits, const T* repetition_penalties, T* output,
                                        const int32_t vocab_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < vocab_size) {
    T l = logits[tid];
    T r = repetition_penalties[tid];
    l = l > (T)0 ? l / r : l * r;
    output[tid] = l;
  }
}

template <typename T>
void InvokeRepetitionPenalty(const T* logits, const T* repetition_penalties, T* output, const int32_t vocab_size,
                             cudaStream_t& stream) {
  dim3 grid((vocab_size + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  RepetitionPenaltyKernel<<<grid, block, 0, stream>>>(logits, repetition_penalties, output, vocab_size);
}

#define INSTANTIATE_INVOKE_REPETITION_PENALTY(T)                                                   \
  template void InvokeRepetitionPenalty(const T* logits, const T* repetition_penalties, T* output, \
                                        const int32_t vocab_size, cudaStream_t& stream);

INSTANTIATE_INVOKE_REPETITION_PENALTY(float);
INSTANTIATE_INVOKE_REPETITION_PENALTY(half);
INSTANTIATE_INVOKE_REPETITION_PENALTY(__nv_bfloat16);

#undef INSTANTIATE_INVOKE_REPETITION_PENALTY

}  // namespace nvidia
}  // namespace llm_kernels
