/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"

#include <fstream>
#include <iostream>
#include <ostream>

#include "csrc/utils/nvidia/cuda_utils.h"

#include "yaml-cpp/yaml.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

static constexpr size_t CUBLASLT_MATMUL_ALGO_BUFFER_LEN = 8ull;

bool GPUGemmAlgoHelper::SaveToYaml(const std::string& yaml_file) {
  std::ofstream helper_file(yaml_file);
  if (!helper_file.good()) {
    return false;
  }

  std::vector<uint64_t> cublas_algo_buffer(CUBLASLT_MATMUL_ALGO_BUFFER_LEN, 0);
  YAML::Node sm_root_node;
  for (const auto& sm_algo_map_it : algo_map_) {
    const uint32_t sm = sm_algo_map_it.first;
    for (const auto& cuda_ver_algo_map_it : sm_algo_map_it.second) {
      const uint32_t cuda_ver = cuda_ver_algo_map_it.first;
      YAML::Node cuda_ver_node;
      for (const auto& algo_map_it : cuda_ver_algo_map_it.second) {
        YAML::Node algo_node;
        const GemmAlgoFingerprint& gemm_algo_fingerprint = algo_map_it.first;
        const GemmAlgoInfo& gemm_algo_info = algo_map_it.second;
        algo_node["batch_size"] = gemm_algo_fingerprint.batch_size;
        algo_node["m"] = gemm_algo_fingerprint.m;
        algo_node["n"] = gemm_algo_fingerprint.n;
        algo_node["k"] = gemm_algo_fingerprint.k;
        algo_node["a_data_type"] = static_cast<int32_t>(gemm_algo_fingerprint.a_data_type);
        algo_node["b_data_type"] = static_cast<int32_t>(gemm_algo_fingerprint.b_data_type);
        algo_node["c_data_type"] = static_cast<int32_t>(gemm_algo_fingerprint.c_data_type);
        algo_node["compute_type"] = static_cast<int32_t>(gemm_algo_fingerprint.compute_type);
        algo_node["trans_a"] = static_cast<int32_t>(gemm_algo_fingerprint.trans_a);
        algo_node["trans_b"] = static_cast<int32_t>(gemm_algo_fingerprint.trans_b);
        algo_node["op_type"] = static_cast<int32_t>(gemm_algo_info.gemm_op_type);
        algo_node["gpu_elapsed_ms"] = gemm_algo_info.gemm_algo_perf.gpu_elapsed_ms;
        if (gemm_algo_info.gemm_op_type == CUBLASLT_GEMM_ALGO) {
          memcpy(cublas_algo_buffer.data(), &(gemm_algo_info.cublaslt_algo), sizeof(cublasLtMatmulAlgo_t));
          algo_node["blas_algo_id"] = cublas_algo_buffer;
        }
        if (gemm_algo_info.gemm_op_type == DEEPGEMM_ALGO) {
          algo_node["smem_size"] = gemm_algo_info.deepgemm_op_config.smem_size;
          algo_node["num_sms"] = gemm_algo_info.deepgemm_op_config.num_sms;
        }
        cuda_ver_node[cuda_ver].push_back(algo_node);
      }
      sm_root_node[sm].push_back(cuda_ver_node);
    }
  }
  helper_file << sm_root_node;
  return true;
}

bool GPUGemmAlgoHelper::LoadFromYaml(const std::string& yaml_file) {
  std::ifstream helper_file(yaml_file.c_str());
  if (!helper_file.good()) {
    return false;
  }

  std::vector<uint64_t> cublas_algo_buffer(CUBLASLT_MATMUL_ALGO_BUFFER_LEN, 0);
  YAML::Node sm_root_node = YAML::LoadFile(yaml_file);
  for (YAML::const_iterator sm_level_it = sm_root_node.begin(); sm_level_it != sm_root_node.end(); ++sm_level_it) {
    const uint32_t sm = sm_level_it->first.as<uint32_t>();
    for (YAML::const_iterator cuda_ver_level_it = sm_root_node[sm].begin(); cuda_ver_level_it != sm_root_node[sm].end();
         ++cuda_ver_level_it) {
      YAML::Node::const_iterator sub_it = cuda_ver_level_it->begin();
      const uint32_t cuda_ver = sub_it->first.as<uint32_t>();

      for (YAML::const_iterator algo_level_it = sub_it->second.begin(); algo_level_it != sub_it->second.end();
           ++algo_level_it) {
        YAML::Node algo_node = algo_level_it->as<YAML::Node>();
        GemmAlgoFingerprint gemm_algo_fingerprint;
        GemmAlgoInfo gemm_algo_info;
        gemm_algo_fingerprint.batch_size = algo_node["batch_size"].as<uint64_t>();
        gemm_algo_fingerprint.m = algo_node["m"].as<uint64_t>();
        gemm_algo_fingerprint.n = algo_node["n"].as<uint64_t>();
        gemm_algo_fingerprint.k = algo_node["k"].as<uint64_t>();
        gemm_algo_fingerprint.a_data_type = static_cast<cudaDataType_t>(algo_node["a_data_type"].as<int32_t>());
        gemm_algo_fingerprint.b_data_type = static_cast<cudaDataType_t>(algo_node["b_data_type"].as<int32_t>());
        gemm_algo_fingerprint.c_data_type = static_cast<cudaDataType_t>(algo_node["c_data_type"].as<int32_t>());
        gemm_algo_fingerprint.compute_type = static_cast<cudaDataType_t>(algo_node["compute_type"].as<int32_t>());
        gemm_algo_fingerprint.trans_a = static_cast<cublasOperation_t>(algo_node["trans_a"].as<int32_t>());
        gemm_algo_fingerprint.trans_b = static_cast<cublasOperation_t>(algo_node["trans_b"].as<int32_t>());
        gemm_algo_info.gemm_op_type = static_cast<GemmOpType>(algo_node["op_type"].as<int32_t>());
        if (gemm_algo_info.gemm_op_type == CUBLASLT_GEMM_ALGO) {
          cublas_algo_buffer = algo_node["blas_algo_id"].as<std::vector<uint64_t>>();
          cublasLtMatmulAlgo_t blas_algo_id;
          memcpy(&blas_algo_id, cublas_algo_buffer.data(), sizeof(cublasLtMatmulAlgo_t));
          gemm_algo_info.cublaslt_algo = blas_algo_id;
        }
        if (gemm_algo_info.gemm_op_type == DEEPGEMM_ALGO) {
          gemm_algo_info.deepgemm_op_config.smem_size = algo_node["smem_size"].as<uint32_t>();
          gemm_algo_info.deepgemm_op_config.num_sms = algo_node["num_sms"].as<uint32_t>();
          gemm_algo_info.deepgemm_op_config.m = algo_node["m"].as<uint32_t>();
          gemm_algo_info.deepgemm_op_config.n = algo_node["n"].as<uint32_t>();
          gemm_algo_info.deepgemm_op_config.k = algo_node["k"].as<uint32_t>();
        }
        AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint, gemm_algo_info);
      }
    }
  }
  return true;
}

bool GPUGemmAlgoHelper::AddGemmAlgo(const uint32_t sm, const uint32_t cuda_ver, GemmAlgoFingerprint fingerprint,
                                    GemmAlgoInfo gemm_algo_info) {
  std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>& algos_map =
      GetOrCreateAlgoMap(sm, cuda_ver);
  algos_map[fingerprint] = gemm_algo_info;
  is_init_ = true;
  return true;
}

const GemmAlgoInfo GPUGemmAlgoHelper::GetGemmAlgo(const uint32_t sm, const uint32_t cuda_ver, const uint64_t batch_size,
                                                  const uint64_t m, const uint64_t n, const uint64_t k,
                                                  const cudaDataType_t a_dtype, const cudaDataType_t b_dtype,
                                                  const cudaDataType_t c_dtype, const cudaDataType_t compute_type,
                                                  const cublasOperation_t trans_a, const cublasOperation_t trans_b) {
  GemmAlgoInfo empty_gemm_algo_info;
  if (!is_init_) {
    return empty_gemm_algo_info;
  }

  const auto& cuda_ver_map_it = algo_map_.find(sm);
  if (cuda_ver_map_it == algo_map_.end()) {
    return empty_gemm_algo_info;
  }

  const auto& all_algos_map_it = cuda_ver_map_it->second.find(cuda_ver);
  if (all_algos_map_it == cuda_ver_map_it->second.end()) {
    return empty_gemm_algo_info;
  }

  GemmAlgoFingerprint fingerprint{batch_size, m, n, k, trans_a, trans_b, a_dtype, b_dtype, c_dtype, compute_type};
  const auto& algo_map_it = all_algos_map_it->second.find(fingerprint);
  if (algo_map_it == all_algos_map_it->second.end()) {
    return empty_gemm_algo_info;
  }

  return all_algos_map_it->second[fingerprint];
}

bool GPUGemmAlgoHelper::IsGemmAlgoExist(const uint32_t sm, const uint32_t cuda_ver, const uint64_t batch_size,
                                        const uint64_t m, const uint64_t n, const uint64_t k,
                                        const cudaDataType_t a_dtype, const cudaDataType_t b_dtype,
                                        const cudaDataType_t c_dtype, const cudaDataType_t compute_type,
                                        const cublasOperation_t trans_a, const cublasOperation_t trans_b) {
  if (!is_init_) {
    return is_init_;
  }
  const auto& cuda_ver_map_it = algo_map_.find(sm);
  if (cuda_ver_map_it == algo_map_.end()) {
    return false;
  }
  const auto& all_algos_map_it = cuda_ver_map_it->second.find(cuda_ver);
  if (all_algos_map_it == cuda_ver_map_it->second.end()) {
    return false;
  }

  GemmAlgoFingerprint fingerprint{batch_size, n, m, k, trans_a, trans_b, a_dtype, b_dtype, c_dtype, compute_type};
  return all_algos_map_it->second.find(fingerprint) != all_algos_map_it->second.end();
}

std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>& GPUGemmAlgoHelper::GetOrCreateAlgoMap(
    const uint32_t sm, const uint32_t cuda_ver) {
  const auto& cuda_ver_algo_map_it = algo_map_.find(sm);
  if (cuda_ver_algo_map_it == algo_map_.end()) {
    std::unordered_map<uint32_t, std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher>>
        cuda_ver_algo_map;
    algo_map_.emplace(sm, std::move(cuda_ver_algo_map));
  }

  const auto& all_algos_map_it = algo_map_[sm].find(cuda_ver);
  if (all_algos_map_it == algo_map_[sm].end()) {
    std::unordered_map<GemmAlgoFingerprint, GemmAlgoInfo, GemmFingerprintHasher> all_algos_map;
    algo_map_[sm].emplace(cuda_ver, std::move(all_algos_map));
  }

  return algo_map_[sm][cuda_ver];
}

uint32_t GetAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}

cublasLtMatmulAlgo_t HeuristicSearchCublasAlgo(cublasLtHandle_t cublaslt_handle, cublasOperation_t transa,
                                               cublasOperation_t transb, const int32_t m, const int32_t n,
                                               const int32_t k, const void* a_ptr, const int32_t lda,
                                               cudaDataType_t a_type, const void* b_ptr, const int32_t ldb,
                                               cudaDataType_t b_type, void* c_ptr, const int32_t ldc,
                                               cudaDataType_t c_type, float f_alpha, float f_beta,
                                               cudaDataType_t compute_type, const size_t workspace_size) {
  std::vector<cublasLtMatmulHeuristicResult_t> algo_result =
      HeuristicSearchCublasAlgo(cublaslt_handle, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb, b_type, c_ptr,
                                ldc, c_type, f_alpha, f_beta, compute_type, workspace_size, /*top_algo_num*/ 1);
  return algo_result.begin()->algo;
}

std::vector<cublasLtMatmulHeuristicResult_t> HeuristicSearchCublasAlgo(
    cublasLtHandle_t cublaslt_handle, cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
    const int32_t n, const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type, const void* b_ptr,
    const int32_t ldb, cudaDataType_t b_type, void* c_ptr, const int32_t ldc, cudaDataType_t c_type, float f_alpha,
    float f_beta, cudaDataType_t compute_type, const size_t workspace_size, const size_t top_algo_num) {
#if (CUBLAS_VERSION) <= 110402
  std::cerr << "cublas version: " << CUBLAS_VERSION << " is too low to support cublas algo search." << std::endl;
  return {};
#endif

  std::vector<cublasLtMatmulHeuristicResult_t> algo_result(top_algo_num);

  // TODO(karlluo): will invoke accuraccy problem
  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? 1 : 0;

  // prepare description
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t inner_compute_type;
  int returned_result = 0;

  if (is_fp16_compute_type) {
    // TODO(karlluo): support CUBLAS_COMPUTE_32F_FAST_TF32
    inner_compute_type = CUBLAS_COMPUTE_16F;
  } else {
    inner_compute_type = CUBLAS_COMPUTE_32F;
  }

  // Create descriptors for the original matrices
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatrixLayoutCreate(&a_desc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatrixLayoutCreate(&b_desc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, ldc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulDescCreate(&operation_desc, inner_compute_type, scale_type));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
  cublasLtMatmulPreference_t preference_desc = nullptr;
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceCreate(&preference_desc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceInit(preference_desc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
  uint32_t pointer_mode_mask = 0;
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(preference_desc, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK,
                                                               &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif
  uint32_t a_alignment = GetAlignment(reinterpret_cast<uintptr_t>(a_ptr));
  uint32_t b_alignment = GetAlignment(reinterpret_cast<uintptr_t>(b_ptr));
  uint32_t c_alignment = GetAlignment(reinterpret_cast<uintptr_t>(c_ptr));
  // TODO(karlluo): support bias
  uint32_t d_alignment = GetAlignment(reinterpret_cast<uintptr_t>(/*bias*/ nullptr));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &a_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &b_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &c_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &d_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operation_desc, a_desc, b_desc, c_desc,
                                                         c_desc, preference_desc, algo_result.size(),
                                                         algo_result.data(), &returned_result));
  if (returned_result == 0) {
    CHECK_NVIDIA_CUDA_ERROR(CUBLAS_STATUS_NOT_SUPPORTED);
  }
  algo_result.resize(returned_result);

  return algo_result;
}

}  // namespace nvidia
}  // namespace llm_kernels
