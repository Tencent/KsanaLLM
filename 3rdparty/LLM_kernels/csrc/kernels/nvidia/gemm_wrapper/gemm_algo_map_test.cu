/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class GPUGemmAlgoHelperTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    CHECK_NVIDIA_CUDA_ERROR(cublasCreate(&cublas_handle));
    CHECK_NVIDIA_CUDA_ERROR(cublasLtCreate(&cublaslt_handle));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&cublas_workspace_buffer_ptr, GetCublasWorkspaceSize()));
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    CHECK_NVIDIA_CUDA_ERROR(cublasDestroy(cublas_handle));
    CHECK_NVIDIA_CUDA_ERROR(cublasLtDestroy(cublaslt_handle));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(cublas_workspace_buffer_ptr));
  }

 protected:
  using NvidiaTestSuitBase::stream;

  void* cublas_workspace_buffer_ptr{nullptr};

  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
};

TEST_F(GPUGemmAlgoHelperTestSuit, GPUGemmAlgoHelperTest) {
  using INPUT_DTYPE = __nv_bfloat16;
  using OUTPUT_DTYPE = __nv_bfloat16;
  uint32_t sm = 86;
  uint32_t cuda_ver = 118;  // CUDA 11.8
  uint64_t batch_size = 1;
  uint64_t m = 32;
  uint64_t n = 32;
  uint64_t k = 32;
  BufferMeta a_buffer =
      CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, m + 1, k + 1}, /*is_random_init*/ true);
  BufferMeta b_buffer =
      CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, k + 1, n + 1}, /*is_random_init*/ true);
  BufferMeta c_buffer =
      CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, m + 1, n + 1}, /*is_random_init*/ false);

  GPUGemmAlgoHelper gpu_gemm_algo_helper;
  GemmAlgoFingerprint gemm_algo_fingerprint_1{batch_size,  m,           n,           k,           CUBLAS_OP_N,
                                              CUBLAS_OP_N, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F};
  int32_t lda = k;
  int32_t ldb = n;
  int32_t ldc = n;
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasLtMatmulAlgo_t cublaslt_algo = HeuristicSearchCublasAlgo(
      cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, b_buffer.data_ptr, ldb, CUDA_R_16BF, a_buffer.data_ptr, lda,
      CUDA_R_16BF, c_buffer.data_ptr, ldc, CUDA_R_16BF, alpha, beta, CUDA_R_32F, GetCublasWorkspaceSize());

  DeleteBuffer(c_buffer);
  DeleteBuffer(b_buffer);
  DeleteBuffer(a_buffer);

  GemmAlgoInfo algo_info_1;
  algo_info_1.gemm_op_type = CUBLASLT_GEMM_ALGO;
  algo_info_1.cublaslt_algo = cublaslt_algo;
  gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint_1, algo_info_1);
  gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint_1, algo_info_1);  // should be unique
  gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint_1, algo_info_1);  // should be unique

  GemmAlgoFingerprint gemm_algo_fingerprint_2{batch_size,  m + 1,       n + 1,       k + 1,       CUBLAS_OP_N,
                                              CUBLAS_OP_N, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F};
  cublaslt_algo =
      HeuristicSearchCublasAlgo(cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n + 1, m + 1, k + 1, b_buffer.data_ptr,
                                ldb + 1, CUDA_R_16BF, a_buffer.data_ptr, lda + 1, CUDA_R_16BF, c_buffer.data_ptr,
                                ldc + 1, CUDA_R_16BF, alpha, beta, CUDA_R_32F, GetCublasWorkspaceSize());
  GemmAlgoInfo algo_info_2;
  algo_info_2.gemm_op_type = CUBLASLT_GEMM_ALGO;
  algo_info_2.cublaslt_algo = cublaslt_algo;
  gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint_2, algo_info_2);  // should be different
  gpu_gemm_algo_helper.SaveToYaml("gemm_algo_map.yaml");

  GPUGemmAlgoHelper gpu_gemm_algo_helper_other;
  EXPECT_TRUE(gpu_gemm_algo_helper_other.LoadFromYaml("gemm_algo_map.yaml"));
  EXPECT_TRUE(gpu_gemm_algo_helper_other.GetOrCreateAlgoMap(sm, cuda_ver)[gemm_algo_fingerprint_1].gemm_op_type ==
              algo_info_1.gemm_op_type);
  char buffer_1[sizeof(cublasLtMatmulAlgo_t)];
  char buffer_2[sizeof(cublasLtMatmulAlgo_t)];
  memcpy(buffer_1,
         &(gpu_gemm_algo_helper_other.GetOrCreateAlgoMap(sm, cuda_ver)[gemm_algo_fingerprint_1].cublaslt_algo),
         sizeof(cublasLtMatmulAlgo_t));
  memcpy(buffer_2, &(algo_info_1.cublaslt_algo), sizeof(cublasLtMatmulAlgo_t));
  EXPECT_TRUE(strcmp(buffer_1, buffer_2) == 0);
  memcpy(buffer_1,
         &(gpu_gemm_algo_helper_other.GetOrCreateAlgoMap(sm, cuda_ver)[gemm_algo_fingerprint_2].cublaslt_algo),
         sizeof(cublasLtMatmulAlgo_t));
  memcpy(buffer_2, &(algo_info_2.cublaslt_algo), sizeof(cublasLtMatmulAlgo_t));
  EXPECT_TRUE(strcmp(buffer_1, buffer_2) == 0);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels