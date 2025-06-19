/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeExpand(const T* input, T* output, const int32_t m, const int32_t expand_num, const int32_t n,
                  const size_t stride, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels