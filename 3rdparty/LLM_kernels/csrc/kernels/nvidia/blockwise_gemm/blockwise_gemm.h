/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void BlockwiseGemmKernel(void* a, float* a_scales, void* b, float* b_scales, void* out, int m, int k, int n,
                         cudaStream_t& stream, void* cutlass_buffer = nullptr, size_t cutlass_buffer_size = 0ul);

}  // namespace nvidia
}  // namespace llm_kernels
