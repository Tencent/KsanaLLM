/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {
namespace dequant {

void dequant_int4_fp8(cudaStream_t stream, void* output, const void* input, size_t datasize);

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels
