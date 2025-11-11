/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/utils/nvidia/cuda_utils.h"
#include "weight_scale_kernel.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void WeightScaleKernel(const T* input_weights, float* q_scale, float n_heads_inv_sqrt, float softmax_scale,
                                  int total_tokens, int n_heads) {
  int token_idx = blockIdx.x;
  int head_idx = threadIdx.x;

  // Since q_scale shape is [total_tokens, n_heads, q_scale_groups] and q_scale_groups = 1
  // The index should be token_idx * n_heads + head_idx, i.e., scale_idx = weights_idx
  int weights_idx = token_idx * n_heads + head_idx;

  // Apply scaling: weights = weights * n_heads**-0.5 * q_scale * softmax_scale
  // Store the result back to q_scale array (reusing memory)
  float scaled_weight =
      static_cast<float>(input_weights[weights_idx]) * n_heads_inv_sqrt * q_scale[weights_idx] * softmax_scale;

  q_scale[weights_idx] = scaled_weight;
}

template <typename T>
void InvokeWeightScale(const T* input_weights, float* q_scale, float n_heads_inv_sqrt, float softmax_scale,
                       int total_tokens, int n_heads, cudaStream_t stream) {
  dim3 grid(total_tokens);
  dim3 block(n_heads);

  WeightScaleKernel<T>
      <<<grid, block, 0, stream>>>(input_weights, q_scale, n_heads_inv_sqrt, softmax_scale, total_tokens, n_heads);
}

// Explicit template instantiations
#define INSTANTIATE_INVOKE_WEIGHT_SCALE(T)                                                           \
  template void InvokeWeightScale<T>(const T* input_weights, float* q_scale, float n_heads_inv_sqrt, \
                                     float softmax_scale, int total_tokens, int n_heads, cudaStream_t stream)

INSTANTIATE_INVOKE_WEIGHT_SCALE(float);
INSTANTIATE_INVOKE_WEIGHT_SCALE(half);
INSTANTIATE_INVOKE_WEIGHT_SCALE(__nv_bfloat16);
#undef INSTANTIATE_INVOKE_WEIGHT_SCALE

}  // namespace nvidia
}  // namespace llm_kernels