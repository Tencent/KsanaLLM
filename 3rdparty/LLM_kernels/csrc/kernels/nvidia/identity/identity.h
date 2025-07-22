/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

namespace llm_kernels {
namespace nvidia {
template <typename T>
void InitIdentityMatrixAdaptive(T* matrix, size_t Rows, size_t Cols, cudaStream_t stream);
}  // namespace nvidia
}  // namespace llm_kernels
