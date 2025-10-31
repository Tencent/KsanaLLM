/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <unordered_map>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeLayerNormWithBeta(T* out, const T* input, const T* gamma, const T* beta, const float layernorm_eps,
                             const int32_t m, const int32_t n, float* scale, float* dynamic_scale,
                             const int32_t int8_mode, cudaStream_t stream, int32_t opt_version = 2);

template <typename T>
void InvokeLayerNormWithBeta(T* out, const T* input, const T* gamma, const T* beta, const float layernorm_eps,
                             const int32_t m, const int32_t n, float* scale, const int32_t int8_mode,
                             cudaStream_t stream, int32_t opt_version = 2) {
  InvokeLayerNormWithBeta(out, input, gamma, beta, layernorm_eps, m, n, scale, (float*)nullptr, int8_mode, stream,
                          opt_version);
}

template <typename T>
void InvokeLayerNorm(T* out, const T* input, const T* gamma, const T* beta, float layernorm_eps, int32_t m, int32_t n,
                     cudaStream_t stream);

template <typename T>
void InvokeRMSNorm(T* out, T* input, T* gamma, float layernorm_eps, int32_t m, int32_t n, bool enable_pdl,
                   cudaStream_t stream);

template <typename T>
void InvokeRmsNorm3D(T* out, const T* input, const T* gamma, const float layernorm_eps, const int32_t total_tokens,
                     const int32_t m, const int32_t n, const int32_t start, const int32_t end, const int64_t* mask,
                     cudaStream_t stream);

template <typename T>
void InvokeFusedQKVRmsNorm(T* out, const T* input, const T* q_gamma, const T* k_gamma, const float layernorm_eps,
                           const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                           const int32_t head_size, const int64_t* mask, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
