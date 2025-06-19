/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
void Concat(const T* __restrict__ input_a, const T* __restrict__ input_b, size_t concat_size_a, size_t concat_size_b,
            size_t outer_dim_size, size_t inner_dim_size, T* __restrict__ output, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
