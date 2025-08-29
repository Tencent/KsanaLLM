/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/adjust_mem/adjust_mem.h"

#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;
namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeExtractSubMatrix(const T* __restrict__ input, T* __restrict__ output, size_t m, size_t input_n,
                            size_t output_n, cudaStream_t& stream) {
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy2DAsync(reinterpret_cast<void*>(output), output_n * sizeof(T),
                                            reinterpret_cast<const void*>(input), input_n * sizeof(T),
                                            output_n * sizeof(T), m, cudaMemcpyDeviceToDevice, stream));
}

#define INSTANTIATE_GATHER_SUBMATRIX(T)                                                                               \
  template void InvokeExtractSubMatrix(const T* __restrict__ input, T* __restrict__ output, size_t m, size_t input_n, \
                                       size_t output_n, cudaStream_t& stream);

INSTANTIATE_GATHER_SUBMATRIX(float);
INSTANTIATE_GATHER_SUBMATRIX(half);
INSTANTIATE_GATHER_SUBMATRIX(__nv_bfloat16);

#undef INSTANTIATE_GATHER_SUBMATRIX

}  // namespace nvidia()
}  // namespace llm_kernels