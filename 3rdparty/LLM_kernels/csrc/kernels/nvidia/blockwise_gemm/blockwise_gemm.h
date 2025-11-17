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

/**
 * @brief Blockwise quantized GEMM kernel function
 *
 * This function implements blockwise quantized matrix multiplication using the CUTLASS library,
 * supporting FP8 quantization format. It performs GEMM computation on quantized matrices with
 * block-level scaling factors, optimized for efficient matrix operations in large language model inference.
 *
 * @param T Output data type, supports half and __nv_bfloat16, does not support float
 * @param a Pointer to input matrix A in FP8 E4M3 quantized format
 * @param a_scales Block-level scaling factors array for matrix A, used for dequantization
 * @param b Pointer to input matrix B in FP8 E4M3 quantized format
 * @param b_scales Block-level scaling factors array for matrix B, used for dequantization
 * @param out Pointer to output matrix storing GEMM computation results
 * @param m Matrix dimension: number of rows in A and output matrix
 * @param k Matrix dimension: number of columns in A and rows in B
 * @param n Matrix dimension: number of columns in B and output matrix
 * @param stream CUDA stream for asynchronous execution
 * @param cutlass_buffer Optional CUTLASS library workspace buffer pointer
 * @param cutlass_buffer_size Size of CUTLASS workspace buffer, default is 0
 *
 * @note This function requires GPU architecture SM90 or higher, otherwise it will skip execution
 * @note When k > 3*n and workspace buffer is provided, StreamKScheduler will be used for better performance
 * @note The function internally dispatches to appropriate CUTLASS GEMM implementation based on data type
 */
template <typename T>
void BlockwiseGemmKernel(void* a, float* a_scales, void* b, float* b_scales, void* out, int m, int k, int n,
                         cudaStream_t& stream, void* cutlass_buffer = nullptr, size_t cutlass_buffer_size = 0ul);

/**
 * @brief Calculate workspace buffer size required for blockwise quantized GEMM kernel
 *
 * This function computes the workspace buffer size needed by the CUTLASS library for
 * blockwise quantized GEMM operations.
 *
 * @param T Output data type, supports half and __nv_bfloat16, does not support float
 * @param m Matrix dimension: number of rows in matrix A and output matrix
 * @param k Matrix dimension: number of columns in matrix A and rows in matrix B
 * @param n Matrix dimension: number of columns in matrix B and output matrix
 * @return size_t Required workspace buffer size in bytes, returns 0 if SM version < 90 or unsupported type
 *
 * @note This function requires GPU architecture SM90 or higher, otherwise returns 0
 * @note The returned size is used to allocate workspace buffer for BlockwiseGemmKernel function
 * @note Workspace size depends on matrix dimensions and CUTLASS internal requirements
 */
template <typename T>
size_t GetBlockwiseGemmWorkspaceSize(int m, int k, int n);
}  // namespace nvidia
}  // namespace llm_kernels
