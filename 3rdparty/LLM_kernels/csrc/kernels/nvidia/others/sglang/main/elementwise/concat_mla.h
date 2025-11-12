/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

// Concatenate k_nope and k_rope with head expansion
template <typename T>
void concat_mla_k(const T* k_nope,  // [num_tokens, num_heads, qk_nope_head_dim]
                  const T* k_rope,  // [num_tokens, 1, qk_rope_head_dim]
                  T* k,             // [num_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
                  const int num_tokens, const int num_heads, const int qk_nope_head_dim, const int qk_rope_head_dim,
                  cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
