/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llm_kernels {
namespace nvidia {

// NOTE(karlluo): default cublas workspace is 32MB
#define DEFAULT_CUBLAS_WORKSPACE_SIZE 33554432
#define DEFAULT_CUBLAS_BATCH_GEMM_SIZE 3

inline size_t GetCublasWorkspaceSize() {
  const char* val = std::getenv("CUBLASLT_WORKSPACE_SIZE");
  size_t workspace_size = 32 * 1024ul;
  if (val) {
    try {
      workspace_size = std::stoi(val);
    } catch (std::invalid_argument const& e) {
      std::cerr << "invalid CUBLASLT_WORKSPACE_SIZE, using default workspace size of " << DEFAULT_CUBLAS_WORKSPACE_SIZE
                << " bytes.";
      return DEFAULT_CUBLAS_WORKSPACE_SIZE;
    } catch (std::out_of_range const& e) {
      std::cerr << "CUBLASLT_WORKSPACE_SIZE out of range, using default workspace size of "
                << DEFAULT_CUBLAS_WORKSPACE_SIZE << " bytes.";
      return DEFAULT_CUBLAS_WORKSPACE_SIZE;
    }
    return workspace_size * 1024ul;
  } else {
    return DEFAULT_CUBLAS_WORKSPACE_SIZE;
  }
}

cublasStatus_t InvokeCublasGemmEx(cublasHandle_t cublas_handle, cublasOperation_t transa, cublasOperation_t transb,
                                  const int32_t m, const int32_t n, const int32_t k, const void* alpha,
                                  const void* a_ptr, cudaDataType_t a_type, int32_t lda, const void* b_ptr,
                                  cudaDataType_t b_type, int32_t ldb, const void* beta, void* c_ptr,
                                  cudaDataType_t c_type, int32_t ldc, cudaDataType_t compute_type,
                                  cublasGemmAlgo_t algo);

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                cudaStream_t& stream, void* workspace_ptr = nullptr, size_t workspace_size = 0);

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                cudaStream_t& stream, void* workspace_ptr, size_t workspace_size,
                                cublasLtMatmulAlgo_t* cublaslt_algo);

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                const int32_t batch_count, cudaStream_t& stream, void* workspace_ptr,
                                size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo);

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, const int32_t batch_count, float f_alpha,
                                float f_beta, cudaDataType_t compute_type, cudaStream_t& stream, void* workspace_ptr,
                                size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo,
                                const void* a_scale = nullptr, const void* b_scale = nullptr);

cublasStatus_t InvokeCublasStridedBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                              cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                              const int32_t n, const int32_t k, const void* A, const int32_t lda,
                                              const int64_t strideA, cudaDataType_t Atype, const void* B,
                                              const int32_t ldb, const int64_t strideB, cudaDataType_t Btype, void* C,
                                              const int32_t ldc, const int64_t strideC, cudaDataType_t Ctype,
                                              const int32_t batch_count, cudaDataType_t compute_type,
                                              const float f_alpha = 1.0f, const float f_beta = 0.0f);

cublasStatus_t InvokeCublasStridedBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                              cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                              const int32_t n, const int32_t k, const float f_alpha, const void* A,
                                              cudaDataType_t AType, const int32_t lda, const int64_t strideA,
                                              const void* B, cudaDataType_t BType, const int32_t ldb,
                                              const int64_t strideB, const float f_beta, void* C, cudaDataType_t CType,
                                              const int32_t ldc, const int64_t strideC, const int32_t batch_count,
                                              cudaDataType_t compute_type);

cublasStatus_t InvokeCublasBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                       cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                       const int32_t n, const int32_t k, const void* const* A, const int32_t lda,
                                       cudaDataType_t AType, const void* const* B, const int32_t ldb,
                                       cudaDataType_t BType, void* const* C, const int32_t ldc, cudaDataType_t CType,
                                       cudaDataType_t compute_type, const int32_t batch_count);

cudaError_t InvokeCustomGemm(cudaStream_t stream, cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                             const int32_t n, const int32_t k, void const* A, const int32_t lda, cudaDataType_t AType,
                             void const* B, const int32_t ldb, cudaDataType_t BType, void* C, const int32_t ldc,
                             cudaDataType_t CType, cudaDataType_t compute_type, const float alpha);

}  // namespace nvidia
}  // namespace llm_kernels
