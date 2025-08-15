/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Apply token bitmask to logits in-place, setting masked tokens to negative infinity
template <typename T>
void ApplyTokenBitmaskInplace(T* logits, const int32_t* bitmask, const int32_t* indices, int32_t vocab_size,
                              int32_t logits_stride, int32_t bitmask_stride, int32_t num_rows, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
