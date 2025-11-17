/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <limits.h>

#include <cfloat>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

enum GemmOpType {
  CUBLASLT_GEMM_ALGO = 0,
  CUSTOM_GEMM_ALGO = 1,
  DEFAULT_GEMM_ALGO = 2,
  CUSTOM_CUTLASS_GEMM_ALGO = 3,
};

constexpr size_t DEFAULT_ALGO_SEARCH_NUM = 200ul;

struct GemmAlgoFingerprint {
  uint64_t batch_size;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  cublasOperation_t trans_a;
  cublasOperation_t trans_b;
  cudaDataType_t a_data_type;
  cudaDataType_t b_data_type;
  cudaDataType_t c_data_type;
  cudaDataType_t compute_type;

  bool operator==(GemmAlgoFingerprint const& config) const {
    return (batch_size == config.batch_size) && (m == config.m) && (n == config.n) && (k == config.k) &&
           (a_data_type == config.a_data_type) && (b_data_type == config.b_data_type) &&
           (c_data_type == config.c_data_type) && (compute_type == config.compute_type) &&
           (trans_a == config.trans_a) && (trans_b == config.trans_b);
  }
};

class GemmFingerprintHasher {
 public:
  std::size_t operator()(GemmAlgoFingerprint const& config) const {
    // NOTE(karlluo): multiplying by a prime number and applying bitwise XOR can serve as a fast hash function within a
    // low-conflict range.
    return config.batch_size * 3597400421ull ^ config.m * 206924999ull ^ config.n * 16547779ull ^
           config.k * 3624845ull ^ static_cast<int32_t>(config.a_data_type) * 3624845ull ^
           static_cast<int32_t>(config.b_data_type) * 149491ull ^ static_cast<int32_t>(config.c_data_type) * 40711ull ^
           static_cast<int32_t>(config.compute_type) * 3047ull ^ static_cast<int32_t>(config.trans_a) * 163ull ^
           static_cast<int32_t>(config.trans_b) * 11ull;
  }
};

struct GemmAlgoPerformance {
  // latency of GEMM operator on GPU in milliseconds, smaller is better
  float gpu_elapsed_ms = FLT_MAX;
  // latency of GEMM operator on host in milliseconds, smaller is better
  float host_elapsed_ms = FLT_MAX;
  // fluctuating noise value of GEMM operator, smaller is better
  float gpu_elapsed_noise_ms = FLT_MAX;
  float host_elapsed_noise_ms = FLT_MAX;
  // GPU HBM bandwith bigger is better
  double global_mem_bandwith = DBL_MIN_EXP;
  // GPU HBM usage bigger is better
  float global_mem_usage = FLT_MIN_EXP;
};

struct GemmAlgoInfo {
  GemmAlgoPerformance gemm_algo_perf;
  GemmOpType gemm_op_type = DEFAULT_GEMM_ALGO;
  // for CUBLASLT
  cublasLtMatmulAlgo_t cublaslt_algo;
};

class GPUGemmAlgoHelper {
 public:
  GPUGemmAlgoHelper(){};
  ~GPUGemmAlgoHelper(){};

  bool SaveToYaml(const std::string& yaml_file);
  bool LoadFromYaml(const std::string& yaml_file);

  bool AddGemmAlgo(const uint32_t sm, const uint32_t cuda_ver, GemmAlgoFingerprint fingerprint,
                   GemmAlgoInfo gemm_algo_info);

  const GemmAlgoInfo GetGemmAlgo(const uint32_t sm, const uint32_t cuda_ver, const uint64_t batch_size,
                                 const uint64_t m, const uint64_t n, const uint64_t k, const cudaDataType_t a_dtype,
                                 const cudaDataType_t b_dtype, const cudaDataType_t c_dtype,
                                 const cudaDataType_t compute_type, const cublasOperation_t trans_a,
                                 const cublasOperation_t trans_b);

  bool IsGemmAlgoExist(const uint32_t sm, const uint32_t cuda_ver, const uint64_t batch_size, const uint64_t m,
                       const uint64_t n, const uint64_t k, const cudaDataType_t a_dtype, const cudaDataType_t b_dtype,
                       const cudaDataType_t c_dtype, const cudaDataType_t compute_type, const cublasOperation_t trans_a,
                       const cublasOperation_t trans_b);

  std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>& GetOrCreateAlgoMap(
      const uint32_t sm, const uint32_t cuda_ver);

  bool IsInit() { return is_init_; }

 private:
  bool is_init_{false};

  // NOTE(karlluo): key: GPU sm, value: { key: CUDA version, value: { key: {b, m, n, k, a_dtype, b_dtype, a_trans,
  // b_trans}, value: cublas_algo / custom_op config } } for example:
  // {
  //   "GPU sm": {
  //       "8.6": {  # Compute Capacity
  //           "CUDA version": {
  //               "11.8": {  # CUDA version
  //                   hash(
  //                       "b": 1,
  //                       "m": 1024,
  //                       "n": 1024,
  //                       "k": 1024,
  //                       "a_dtype": "float16",
  //                       "b_dtype": "float16",
  //                       "a_trans": False,
  //                       "b_trans": True
  //                   ) = hash_key:
  //                               {  # optimized GEMM config
  //                                  # only for cublas
  //                                  "cublas_algo": 12,
  //                                  # only for Custom GEMM kernel
  //                                  "custom_op_config": {
  //                                      "block_size": 128,
  //                                      "tile_size": 32,
  //                                      "use_shared_memory": True
  //                                  }
  //                               }
  //               }
  //           }
  //       }
  //   }
  // }
  std::unordered_map<
      uint32_t,
      std::unordered_map<uint32_t, std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>>>
      algo_map_;
};

// search the best cublas gemm algo
cublasLtMatmulAlgo_t HeuristicSearchCublasAlgo(cublasLtHandle_t cublaslt_handle, cublasOperation_t transa,
                                               cublasOperation_t transb, const int32_t m, const int32_t n,
                                               const int32_t k, const void* a_ptr, const int32_t lda,
                                               cudaDataType_t a_type, const void* b_ptr, const int32_t ldb,
                                               cudaDataType_t b_type, void* c_ptr, const int32_t ldc,
                                               cudaDataType_t c_type, float f_alpha, float f_beta,
                                               cudaDataType_t compute_type, const size_t workspace_size);

// search set of cublas gemm algos
std::vector<cublasLtMatmulHeuristicResult_t> HeuristicSearchCublasAlgo(
    cublasLtHandle_t cublaslt_handle, cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
    const int32_t n, const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type, const void* b_ptr,
    const int32_t ldb, cudaDataType_t b_type, void* c_ptr, const int32_t ldc, cudaDataType_t c_type, float f_alpha,
    float f_beta, cudaDataType_t compute_type, const size_t workspace_size, const size_t top_algo_num);

}  // namespace nvidia
}  // namespace llm_kernels
