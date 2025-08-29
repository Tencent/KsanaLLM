/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

 #pragma once

 #include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeExtractSubMatrix(const T* __restrict__ input, T* __restrict__ output, size_t m, size_t input_n,
                            size_t output_n, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels