/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef ENABLE_BF16
#  include <cuda_bf16.h>
#endif

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Apply weight scaling operation: weights = weights * n_heads**-0.5 * q_scale * softmax_scale
 *
 * This kernel implements the weight scaling operation from the Python reference:
 * weights = weights * self.n_heads**-0.5 * q_scale * self.softmax_scale
 *
 * @param input_weights Input weights tensor [total_tokens, n_heads] (T type)
 * @param q_scale Q scale tensor from FP8 quantization [total_tokens * n_heads], reuse memory of input_weights
 * @param n_heads_inv_sqrt Pre-computed n_heads**-0.5
 * @param softmax_scale Pre-computed softmax scale
 * @param total_tokens Number of tokens
 * @param n_heads Number of heads
 * @param stream CUDA stream
 */
template <typename T>
void InvokeWeightScale(const T* input_weights, float* q_scale, float n_heads_inv_sqrt, float softmax_scale,
                       int total_tokens, int n_heads, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels