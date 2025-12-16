/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"

#include <iostream>
#include <ostream>

#include <cub/cub.cuh>

#include "cutlass/numeric_conversion.h"

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_type_conversion.h"
#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

cublasStatus_t InvokeCublasGemmEx(cublasHandle_t cublas_handle, cublasOperation_t transa, cublasOperation_t transb,
                                  const int32_t m, const int32_t n, const int32_t k, const void* alpha,
                                  const void* a_ptr, cudaDataType_t a_type, int32_t lda, const void* b_ptr,
                                  cudaDataType_t b_type, int32_t ldb, const void* beta, void* c_ptr,
                                  cudaDataType_t c_type, int32_t ldc, cudaDataType_t compute_type,
                                  cublasGemmAlgo_t algo) {
  return cublasGemmEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_type, lda, b_ptr, b_type, ldb, beta,
                      c_ptr, c_type, ldc, compute_type, algo);
}

cublasStatus_t InvokeCustomGemm(cudaStream_t& stream, cublasOperation_t transa, cublasOperation_t transb,
                                const int32_t m, const int32_t n, const int32_t k, const void* a_ptr,
                                const int32_t lda, cudaDataType_t a_type, const void* b_ptr, const int32_t ldb,
                                cudaDataType_t b_type, void* c_ptr, const int32_t ldc, cudaDataType_t c_type,
                                cudaDataType_t compute_type) {
  cudaError_t result = InvokeCustomGemm(stream, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb, b_type, c_ptr,
            ldc, c_type, compute_type, 1.0f);
  if (result == cudaSuccess) {
    return CUBLAS_STATUS_SUCCESS;
  } else if (result == cudaErrorNotSupported) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  } else {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                cudaStream_t& stream, void* workspace_ptr, size_t workspace_size) {
  cublasLtMatmulAlgo_t* cublaslt_algo = nullptr;
  return InvokeCublasGemm(cublas_handle, cublaslt_handle, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb,
                          b_type, c_ptr, ldc, c_type, /*batch_size */ 1, /*f_alpha*/ 1.0f, /*f_beta*/ 0.0f,
                          compute_type, stream, workspace_ptr, workspace_size, cublaslt_algo);
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                cudaStream_t& stream, void* workspace_ptr, size_t workspace_size,
                                cublasLtMatmulAlgo_t* cublaslt_algo, bool use_fp16_compute_reduction) {
  return InvokeCublasGemm(cublas_handle, cublaslt_handle, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb,
                          b_type, c_ptr, ldc, c_type, /*batch_size */ 1, /*f_alpha*/ 1.0f, /*f_beta*/ 0.0f,
                          compute_type, stream, workspace_ptr, workspace_size, cublaslt_algo, nullptr, nullptr,
                          use_fp16_compute_reduction);
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                const int32_t batch_count, cudaStream_t& stream, void* workspace_ptr,
                                size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo) {
  return InvokeCublasGemm(cublas_handle, cublaslt_handle, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb,
                          b_type, c_ptr, ldc, c_type, batch_count, /*f_alpha*/ 1.0f, /*f_beta*/ 0.0f, compute_type,
                          stream, workspace_ptr, workspace_size, cublaslt_algo, nullptr, nullptr, 0, 0, 0,
                          false);
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, const int32_t batch_count, float f_alpha,
                                float f_beta, cudaDataType_t compute_type, cudaStream_t& stream, void* workspace_ptr,
                                size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo, const void* a_scale,
                                const void* b_scale, int64_t batch_offset_a, int64_t batch_offset_b,
                                int64_t batch_offset_c, bool use_fp16_compute_reduction) {
  // NOTE(karlluo): half no static cast in regular c_ptr++
  half h_alpha = (half)(f_alpha);
  half h_beta = (half)(f_beta);
  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? 1 : 0;
  const void* alpha = is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t compute_reduction_type;
  if (is_fp16_compute_type) {
    compute_reduction_type = CUBLAS_COMPUTE_16F;
  } else {
    compute_reduction_type = CUBLAS_COMPUTE_32F;
  }

  // Create descriptors for the original matrices
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&a_desc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&b_desc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, ldc));

  if (batch_count > 1) {
    // Batch is the first dim by default
    if (batch_offset_a == 0) {
      batch_offset_a = m * k;
    }
    if (batch_offset_b == 0) {
      batch_offset_b = k * n;
    }
    if (batch_offset_c == 0) {
      batch_offset_c = m * n;
    }

    cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

    cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_a,
                                     sizeof(batch_offset_a));
    cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_b,
                                     sizeof(batch_offset_b));
    cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset_c,
                                     sizeof(batch_offset_c));
  }

  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescCreate(&operation_desc, compute_reduction_type, scale_type));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
  if (a_scale != nullptr) {
    RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                              &a_scale, sizeof(a_scale)));
  }
  if (b_scale != nullptr) {
    RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                              &b_scale, sizeof(b_scale)));
  }
  workspace_size = (workspace_ptr == nullptr ? 0ul : workspace_size);
  bool is_use_cublaslt_algo = (cublaslt_algo != nullptr) && (workspace_size > 0);

  if (a_type == CUDA_R_16F && use_fp16_compute_reduction) {
    compute_reduction_type = CUBLAS_COMPUTE_16F;
    half h_alpha = static_cast<half>(f_alpha);
    half h_beta = static_cast<half>(f_beta);
    void* alpha = reinterpret_cast<void*>(&h_alpha);
    void* beta = reinterpret_cast<void*>(&h_beta);
    RETURN_NVIDIA_CUBLAS_ERROR(cublasGemmEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_type, lda, b_ptr,
                                          b_type, ldb, beta, c_ptr, c_type, ldc, compute_reduction_type,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmul(cublaslt_handle, operation_desc, alpha, a_ptr, a_desc, b_ptr, b_desc, beta, c_ptr, c_desc, c_ptr,
                     c_desc, is_use_cublaslt_algo ? nullptr : cublaslt_algo, workspace_ptr, workspace_size, stream));
  }

  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescDestroy(operation_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc));

  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t InvokeCublasStridedBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                              cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                              const int32_t n, const int32_t k, const void* a_ptr, const int32_t lda,
                                              const int64_t strideA, cudaDataType_t a_type, const void* b_ptr,
                                              const int32_t ldb, const int64_t strideB, cudaDataType_t b_type,
                                              void* c_ptr, const int32_t ldc, const int64_t strideC,
                                              cudaDataType_t c_type, const int32_t batch_count,
                                              cudaDataType_t compute_type, const float f_alpha, const float f_beta) {
  half h_alpha = (half)f_alpha;
  half h_beta = (half)f_beta;

  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? true : false;
  const void* alpha =
      is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);

  return cublasGemmStridedBatchedEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_type, lda, strideA, b_ptr,
                                    b_type, ldb, strideB, beta, c_ptr, c_type, ldc, strideC, batch_count, compute_type,
                                    CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t InvokeCublasBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                       cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                       const int32_t n, const int32_t k, const void* const* a_ptr, const int32_t lda,
                                       cudaDataType_t a_dtype, const void* const* b_ptr, const int32_t ldb,
                                       cudaDataType_t b_dtype, void* const* c_ptr, const int32_t ldc,
                                       cudaDataType_t c_dtype, cudaDataType_t compute_type, const int32_t batch_count) {
  float f_alpha = static_cast<float>(1.0f);
  float f_beta = static_cast<float>(0.0f);

  half h_alpha = (half)1.0f;
  half h_beta = (half)0.0f;

  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? true : false;
  const void* alpha = is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

  return cublasGemmBatchedEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_dtype, lda, b_ptr, b_dtype, ldb,
                             beta, c_ptr, c_dtype, ldc, batch_count, compute_type, CUBLAS_GEMM_DEFAULT);
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
__global__ void CustomSmallMGemmWithFP32ComputeType(INPUT_DTYPE const* __restrict__ act,
                                                    INPUT_DTYPE const* __restrict__ weight, float alpha,
                                                    OUTPUT_DTYPE* __restrict__ output, int32_t m, int32_t n,
                                                    int32_t k) {
  using VecType = int4;
  static constexpr int32_t k_step = static_cast<int32_t>(128 / (8 * sizeof(INPUT_DTYPE)));
  static constexpr int32_t k_tile = k_step * BLOCK_SIZE;
  auto tile_m_idx = static_cast<int32_t>(blockIdx.x * TILE_M);
  auto tile_n_idx = static_cast<int32_t>(blockIdx.y * TILE_N);
  auto tid = static_cast<int32_t>(threadIdx.x);
  float tile_a[k_step], tile_w[TILE_N * k_step];
  float acc[TILE_M * TILE_N];

  static_assert(k_step % 4 == 0);
  using CVT_INPUT_DTYPE = typename llm_kernels::nvidia::CutlassToTllmTypeAdapter<INPUT_DTYPE>::type;
  // convert input data_type to float with for elemts once
  using Converter = cutlass::NumericArrayConverter<float, CVT_INPUT_DTYPE, 4>;
  using CVT_SRC_DTYPE = typename Converter::source_type;
  using CVT_RES_DTYPE = typename Converter::result_type;
  static constexpr int32_t k_cvt_count = static_cast<int32_t>(sizeof(VecType) / sizeof(CVT_SRC_DTYPE));

#pragma unroll
  for (int32_t i = 0; i < TILE_M * TILE_N; ++i) {
    acc[i] = 0;
  }
  act += tile_m_idx * k;
  weight += tile_n_idx * k;
  output += tile_m_idx * n + tile_n_idx;

  for (int32_t k_idx = tid * k_step; k_idx < k; k_idx += k_tile) {
    for (int32_t i = 0; i < TILE_N; ++i) {
      auto tile_w_quantized = reinterpret_cast<VecType const*>(weight + i * k + k_idx)[0];
#pragma unroll
      for (int32_t cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx) {
        reinterpret_cast<CVT_RES_DTYPE*>(tile_w)[i * k_cvt_count + cvt_idx] =
            Converter::convert(reinterpret_cast<CVT_SRC_DTYPE*>(&tile_w_quantized)[cvt_idx]);
      }
    }
#pragma unroll
    for (int32_t i = 0; i < TILE_M; ++i) {
      auto tile_a_quantized = reinterpret_cast<VecType const*>(act + i * k + k_idx)[0];
#pragma unroll
      for (int32_t cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx) {
        reinterpret_cast<CVT_RES_DTYPE*>(tile_a)[cvt_idx] =
            Converter::convert(reinterpret_cast<CVT_SRC_DTYPE*>(&tile_a_quantized)[cvt_idx]);
      }
#pragma unroll
      for (int32_t j = 0; j < TILE_N; ++j) {
#pragma unroll
        for (int32_t l = 0; l < k_step; ++l) {
          acc[i * TILE_N + j] = fma(tile_a[l], tile_w[j * k_step + l], acc[i * TILE_N + j]);
        }
      }
    }
  }

  typedef cub::WarpReduce<float> WarpReduce;

  static constexpr int32_t WARP_SIZE = 32;
  static constexpr int32_t WARP_NUM = BLOCK_SIZE / WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE;
  __shared__ float shmem[TILE_M * TILE_N * WARP_NUM];
  __shared__ typename WarpReduce::TempStorage tmp_storage[WARP_NUM];
#pragma unroll
  for (int32_t mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
    for (int32_t ni = 0; ni < TILE_N; ++ni) {
      float val = WarpReduce(tmp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
      if (lane_id == 0) {
        shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
      }
    }
  }
  __syncthreads();
  for (int32_t ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
    int32_t mid = ii / TILE_N, nid = ii % TILE_N;
    float val = 0;
#pragma unroll
    for (int32_t jj = 0; jj < WARP_NUM; ++jj) {
      val += shmem[jj * TILE_M * TILE_N + ii];
    }
    output[mid * n + nid] = static_cast<OUTPUT_DTYPE>(val * alpha);
  }
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
void CustomGemmWithFP32ComputeTypeKernel(const int32_t m, const int32_t n, const int32_t k, void const* A,
                                         void const* B, void* C, const float alpha, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(m / TILE_M, n / TILE_N);
  CustomSmallMGemmWithFP32ComputeType<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(reinterpret_cast<INPUT_DTYPE const*>(A), reinterpret_cast<INPUT_DTYPE const*>(B),
                                   alpha, reinterpret_cast<OUTPUT_DTYPE*>(C), m, n, k);
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool CustomSmallMGemmWithFP32ComputeTypeCaller(const int32_t m, const int32_t n, const int32_t k, void const* A,
                                               void const* B, void* C, const float alpha, cudaStream_t stream) {
  constexpr int TILE_M_MAX = 16;
  if (m == TILE_M) {
    CustomGemmWithFP32ComputeTypeKernel<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M, TILE_N, BLOCK_SIZE>(m, n, k, A, B, C, alpha,
                                                                                               stream);
    return true;
  }
  if constexpr (TILE_M < TILE_M_MAX) {
    return CustomSmallMGemmWithFP32ComputeTypeCaller<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M + 1, TILE_N, BLOCK_SIZE>(
        m, n, k, A, B, C, alpha, stream);
  }
  return false;
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
bool CustomSmallMGemmWithFP32ComputeTypeLauncher(const int32_t m, const int32_t n, const int32_t k, void const* A,
                                                 void const* B, void* C, const float alpha, cudaStream_t stream) {
  return CustomSmallMGemmWithFP32ComputeTypeCaller<INPUT_DTYPE, OUTPUT_DTYPE, 1, 2, 128>(m, n, k, A, B, C, alpha,
                                                                                         stream);
}

cudaError_t InvokeCustomSmallMGemmWithFP32ComputeType(cudaStream_t stream, cublasOperation_t transa,
                                                      cublasOperation_t transb, const int32_t m, const int32_t n,
                                                      const int32_t k, void const* A, const int32_t lda,
                                                      cudaDataType_t a_dtype, void const* B, const int32_t ldb,
                                                      cudaDataType_t b_dtype, void* C, const int32_t ldc,
                                                      cudaDataType_t c_dtype, const float alpha) {
  bool is_launch_success = false;
  if (a_dtype == CUDA_R_16F) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_16F) {
      is_launch_success = CustomSmallMGemmWithFP32ComputeTypeLauncher<half, half>(m, n, k, A, B, C, alpha, stream);
    } else if (c_dtype == CUDA_R_32F) {
      is_launch_success = CustomSmallMGemmWithFP32ComputeTypeLauncher<half, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else if (a_dtype == CUDA_R_16BF) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_16BF) {
      is_launch_success =
          CustomSmallMGemmWithFP32ComputeTypeLauncher<__nv_bfloat16, __nv_bfloat16>(m, n, k, A, B, C, alpha, stream);
    } else if (c_dtype == CUDA_R_32F) {
      is_launch_success =
          CustomSmallMGemmWithFP32ComputeTypeLauncher<__nv_bfloat16, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else if (a_dtype == CUDA_R_32F) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_32F) {
      is_launch_success = CustomSmallMGemmWithFP32ComputeTypeLauncher<float, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else {
    std::cerr << "Not support input type: " << a_dtype << std::endl;
    return cudaErrorNotSupported;
  }

  if (is_launch_success) {
    return cudaSuccess;
  } else {
    return cudaErrorLaunchFailure;
  }
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
__global__ void CustomSmallMGemm(INPUT_DTYPE const* __restrict__ act, INPUT_DTYPE const* __restrict__ weight,
                                 float alpha, OUTPUT_DTYPE* __restrict__ output, int32_t m, int32_t n, int32_t k) {
  // TODO(karlluo): async copy and pingpong buffer
  using VecType = int4;
  static constexpr int32_t k_step = static_cast<int32_t>(128 / (8 * sizeof(INPUT_DTYPE)));
  static constexpr int32_t k_tile = k_step * BLOCK_SIZE;
  auto tile_m_idx = static_cast<int32_t>(blockIdx.x * TILE_M);
  auto tile_n_idx = static_cast<int32_t>(blockIdx.y * TILE_N);
  auto tid = static_cast<int32_t>(threadIdx.x);
  float acc[TILE_M * TILE_N];

  static_assert(k_step % 4 == 0);

  vec_t<INPUT_DTYPE, k_step> act_vec;
  vec_t<INPUT_DTYPE, k_step> weight_vec;

#pragma unroll
  for (int32_t i = 0; i < TILE_M * TILE_N; ++i) {
    acc[i] = 0;
  }
  act += tile_m_idx * k;
  weight += tile_n_idx * k;
  output += tile_m_idx * n + tile_n_idx;

  for (int32_t k_idx = tid * k_step; k_idx < k; k_idx += k_tile) {
#pragma unroll
    for (int32_t i = 0; i < TILE_M; ++i) {
      act_vec.load(act + i * k + k_idx);
#pragma unroll
      for (int32_t j = 0; j < TILE_N; ++j) {
        weight_vec.load(weight + j * k + k_idx);
#pragma unroll
        for (int32_t l = 0; l < k_step; ++l) {
          acc[i * TILE_N + j] = fma(act_vec[l], weight_vec[l], acc[i * TILE_N + j]);
        }
      }
    }
  }

  typedef cub::WarpReduce<float> WarpReduce;

  static constexpr int32_t WARP_SIZE = 32;
  static constexpr int32_t WARP_NUM = BLOCK_SIZE / WARP_SIZE;
  int32_t warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE;
  __shared__ OUTPUT_DTYPE shmem[TILE_M * TILE_N * WARP_NUM];
  __shared__ typename WarpReduce::TempStorage tmp_storage[WARP_NUM];
#pragma unroll
  for (int32_t mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
    for (int32_t ni = 0; ni < TILE_N; ++ni) {
      float val = WarpReduce(tmp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
      if (lane_id == 0) {
        shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
      }
    }
  }
  __syncthreads();
  for (int32_t ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
    int32_t mid = ii / TILE_N, nid = ii % TILE_N;
    float val = 0;
#pragma unroll
    for (int32_t jj = 0; jj < WARP_NUM; ++jj) {
      val += static_cast<float>(shmem[jj * TILE_M * TILE_N + ii]);
    }
    output[mid * n + nid] = static_cast<OUTPUT_DTYPE>(val * alpha);
  }
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int32_t TILE_M, int32_t TILE_N, int32_t BLOCK_SIZE>
void CustomSmallMGemmKernel(const int32_t m, const int32_t n, const int32_t k, void const* A, void const* B, void* C,
                            const float alpha, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(m / TILE_M, n / TILE_N);
  CustomSmallMGemm<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(reinterpret_cast<INPUT_DTYPE const*>(A), reinterpret_cast<INPUT_DTYPE const*>(B),
                                   alpha, reinterpret_cast<OUTPUT_DTYPE*>(C), m, n, k);
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE, int TILE_M, int TILE_N, int BLOCK_SIZE>
bool CustomSmallMGemmCaller(const int32_t m, const int32_t n, const int32_t k, void const* A, void const* B, void* C,
                            const float alpha, cudaStream_t stream) {
  constexpr int TILE_M_MAX = 16;
  if (m == TILE_M) {
    CustomSmallMGemmKernel<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M, TILE_N, BLOCK_SIZE>(m, n, k, A, B, C, alpha, stream);
    return true;
  }
  if constexpr (TILE_M < TILE_M_MAX) {
    return CustomSmallMGemmCaller<INPUT_DTYPE, OUTPUT_DTYPE, TILE_M + 1, TILE_N, BLOCK_SIZE>(m, n, k, A, B, C, alpha,
                                                                                             stream);
  }
  return false;
}

template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
bool CustomSmallMGemmLauncher(const int32_t m, const int32_t n, const int32_t k, void const* A, void const* B, void* C,
                              const float alpha, cudaStream_t stream) {
  return CustomSmallMGemmCaller<INPUT_DTYPE, OUTPUT_DTYPE, 1, 2, 128>(m, n, k, A, B, C, alpha, stream);
}

cudaError_t InvokeCustomSmallMGemm(cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb,
                                   const int32_t m, const int32_t n, const int32_t k, void const* A, const int32_t lda,
                                   cudaDataType_t a_dtype, void const* B, const int32_t ldb, cudaDataType_t b_dtype,
                                   void* C, const int32_t ldc, cudaDataType_t c_dtype, const float alpha) {
  bool is_launch_success = false;
  if (a_dtype == CUDA_R_16F) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_16F) {
      is_launch_success = CustomSmallMGemmLauncher<half, half>(m, n, k, A, B, C, alpha, stream);
    } else if (c_dtype == CUDA_R_32F) {
      is_launch_success = CustomSmallMGemmLauncher<half, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else if (a_dtype == CUDA_R_16BF) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_16BF) {
      is_launch_success = CustomSmallMGemmLauncher<__nv_bfloat16, __nv_bfloat16>(m, n, k, A, B, C, alpha, stream);
    } else if (c_dtype == CUDA_R_32F) {
      is_launch_success = CustomSmallMGemmLauncher<__nv_bfloat16, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else if (a_dtype == CUDA_R_32F) {
    if (k % 8 != 0) {
      std::cerr << "Custom gemm not support k % 8 != 0" << std::endl;
      return cudaErrorNotSupported;
    }
    if (c_dtype == CUDA_R_32F) {
      is_launch_success = CustomSmallMGemmLauncher<float, float>(m, n, k, A, B, C, alpha, stream);
    } else {
      std::cerr << "Not support output type: " << c_dtype << std::endl;
      return cudaErrorNotSupported;
    }
  } else {
    std::cerr << "Not support input type: " << a_dtype << std::endl;
    return cudaErrorNotSupported;
  }

  if (is_launch_success) {
    return cudaSuccess;
  } else {
    return cudaErrorLaunchFailure;
  }
}

cudaError_t InvokeCustomGemm(cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                             const int32_t n, const int32_t k, void const* A, const int32_t lda, cudaDataType_t a_dtype,
                             void const* B, const int32_t ldb, cudaDataType_t b_dtype, void* C, const int32_t ldc,
                             cudaDataType_t c_dtype, cudaDataType_t compute_type, const float alpha) {
  if (transa != CUBLAS_OP_N || transb != CUBLAS_OP_T) {
    std::cerr << "Not support A != NoneTrans or B != Trans" << std::endl;
    return cudaErrorNotSupported;
  } else if (n % 2 != 0 || a_dtype != b_dtype) {
    std::cerr << "Not support n % 2 != 0 || a_dtype != b_dtype" << std::endl;
    return cudaErrorNotSupported;
  }

  if (compute_type == CUDA_R_32F) {
    return InvokeCustomSmallMGemmWithFP32ComputeType(stream, transa, transb, m, n, k, A, lda, a_dtype, B, ldb, b_dtype,
                                                     C, ldc, c_dtype, alpha);
  } else if (compute_type == c_dtype) {
    return InvokeCustomSmallMGemm(stream, transa, transb, m, n, k, A, lda, a_dtype, B, ldb, b_dtype, C, ldc, c_dtype,
                                  alpha);
  } else {
    std::cerr << "Not support compute type: " << compute_type << std::endl;
    return cudaErrorNotSupported;
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
