/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "identity.h"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

// CUDA kernel to initialize identity matrix
template <typename T>
__global__ void InitIdentityMatrixKernel(T* matrix, size_t row_size, size_t col_size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < row_size && col < col_size) {
    float val = static_cast<float>(row == col);
    matrix[row * col_size + col] = CastCudaDataType<T>(val);
  }
}

template <typename T>
__global__ void InitDiagonalMatrixKernel(T* matrix, size_t leading_dim, size_t total_elements) {
  int diag_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t matrix_idx = diag_idx * leading_dim + diag_idx;
  if (matrix_idx < total_elements) {
    matrix[matrix_idx] = CastCudaDataType<T>(1.0f);
  }
}

template <typename T>
void InitIdentityMatrixAdaptive(T* matrix, size_t row_size, size_t col_size, cudaStream_t stream) {
  // Threshold for H20
  constexpr size_t THRESHOLD = 2048 * 2048;

  if (row_size * col_size < THRESHOLD) {
    dim3 block_size(16, 16);
    dim3 grid_size((col_size + block_size.x - 1) / block_size.x, (row_size + block_size.y - 1) / block_size.y);
    InitIdentityMatrixKernel<<<grid_size, block_size, 0, stream>>>(matrix, row_size, col_size);
  } else {
    size_t total_elements = row_size * col_size;
    cudaMemsetAsync(matrix, 0, total_elements * sizeof(T), stream);
    dim3 block_size(256);
    dim3 grid_size((total_elements + block_size.x - 1) / block_size.x);
    InitDiagonalMatrixKernel<<<grid_size, block_size, 0, stream>>>(matrix, row_size, total_elements);
  }
}
// Explicit instantiation for float, fp16, bf16, uint8
template void InitIdentityMatrixAdaptive(float*, size_t, size_t, cudaStream_t);
template void InitIdentityMatrixAdaptive(half*, size_t, size_t, cudaStream_t);
template void InitIdentityMatrixAdaptive(__nv_bfloat16*, size_t, size_t, cudaStream_t);

}  // namespace nvidia
}  // namespace llm_kernels
