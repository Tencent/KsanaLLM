/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
template <typename T>
__global__ void CopyElementsKernel(T** src_ptrs, T* dest, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    dest[idx] = *src_ptrs[idx];
  }
}

template <typename T>
void InvokeCopyElements(T** src_ptrs, T* dest, size_t num_elements, cudaStream_t& stream) {
  if (num_elements != 0) {
    dim3 grid((num_elements + llm_kernels::utils::DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) /
              llm_kernels::utils::DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
    dim3 block(llm_kernels::utils::DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
    CopyElementsKernel<<<grid, block, 0, stream>>>(src_ptrs, dest, num_elements);
  }
}

#define INSTANTIATE_INVOKE_COPY_ELEMENTS(T) \
  template void InvokeCopyElements(T** src_ptrs, T* dest, size_t num_elements, cudaStream_t& stream);

INSTANTIATE_INVOKE_COPY_ELEMENTS(float);
INSTANTIATE_INVOKE_COPY_ELEMENTS(half);
INSTANTIATE_INVOKE_COPY_ELEMENTS(__nv_bfloat16);
}  // namespace nvidia
}  // namespace llm_kernels