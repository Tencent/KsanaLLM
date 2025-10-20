/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {
namespace dequant {

void dequant_uint4_fp8_launcher(cudaStream_t stream, void* output, const void* input, size_t datasize);

void dequant_int4_fp8_launcher(cudaStream_t stream, void* output, const void* input, size_t datasize);

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels
