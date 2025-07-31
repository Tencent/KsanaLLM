/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */
#include <limits>
#include <sstream>

#include "gflags/gflags.h"

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

DEFINE_uint64(max_batch_size, 0ull, "max batch size to search");
DEFINE_uint64(max_m, 0ull, "max m to search");
DEFINE_uint64(max_n, 0ull, "max n to search");
DEFINE_uint64(max_k, 0ull, "max k to search");

DEFINE_uint64(batch_size, 0ull, "batch size to search, priority is higher than max_batch_size");
DEFINE_uint64(m, 0ull, "m to search, priority is higher than max_m");
DEFINE_uint64(n, 0ull, "n to search, priority is higher than max_n");
DEFINE_uint64(k, 0ull, "k to search, priority is higher than max_k");

DEFINE_string(input_dtype, "fp16", "input data type, default is fp16, option: fp32, fp16, bf16, fp8_e4m3");
DEFINE_string(output_dtype, "fp16", "output data type, default is fp16, option: fp32, fp16, bf16, fp8_e4m3");
DEFINE_string(inner_compute_dtype, "fp32",
              "inner compute buffer data type, default is fp16, option: fp32, fp16, bf16, fp8_e4m3");

DEFINE_string(output_file, "gemm_algo_map.yaml", "output the search result");

DEFINE_bool(input_a_transop, false,
            "whether input A is tranposed, default is false means CUBLAS_OP_N, true means CUBLAS_OP_T");
DEFINE_bool(input_b_transop, false,
            "whether input B is tranposed, default is false means CUBLAS_OP_N, true means CUBLAS_OP_T");
DEFINE_uint64(op_type, 0ull, "default 0 refer to cuBlas");

DEFINE_bool(append_mode, true, "whether appending algo result to original output file");

void SetSearchRange(uint64_t& left_val, uint64_t& right_val, const std::string& tag_name, uint64_t spec_val,
                    uint64_t max_val) {
  left_val = spec_val != 0 ? spec_val : 1;
  right_val = spec_val != 0 ? spec_val : max_val;
  if (left_val > right_val) {
    std::stringstream ss;
    ss << "max_" << tag_name << ": " << max_val << " " << tag_name << ": " << spec_val
       << " both them is 0, please set one of them for example: --" << tag_name << "=4";
    throw std::invalid_argument(ss.str());
  }
}

cudaDataType_t SetSearchDataType(const std::string& type_flag) {
  if (type_flag == "fp16") {
    return CUDA_R_16F;
  } else if (type_flag == "fp32") {
    return CUDA_R_32F;
  } else if (type_flag == "bf16") {
    return CUDA_R_16BF;
  } else if (type_flag == "fp8_e4m3") {
    return CUDA_R_8F_E4M3;
  } else {
    std::cerr << "Data type " << type_flag << " is not supported." << std::endl;
    exit(0);
  }
  return CUDA_R_32F;
}

cublasOperation_t SetInputTransop(bool type_flag) {
  if (type_flag) {
    return CUBLAS_OP_T;
  } else {
    return CUBLAS_OP_N;
  }
}

float GetCublasGPUElapsedTime(const int warmup_rounds, const int tested_rounds, cublasHandle_t& cublas_handle,
                              cublasLtHandle_t cublaslt_handle, cublasOperation_t trans_b, cublasOperation_t trans_a,
                              uint64_t n_val, uint64_t m_val, uint64_t k_val, void* b_buffer, int32_t ldb,
                              cudaDataType_t b_data_type, void* a_buffer, int32_t lda, cudaDataType_t a_data_type,
                              void* c_buffer, int32_t ldc, cudaDataType_t c_data_type, float alpha, float beta,
                              cudaDataType_t compute_type, cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop,
                              void* workspace_ptr = nullptr, size_t workspace_size = 0,
                              cublasLtMatmulAlgo_t* cublaslt_algo = nullptr) {
  constexpr int batch_count = 1;
  auto cuda_run = [&]() {
    CHECK_NVIDIA_CUDA_ERROR(llm_kernels::nvidia::InvokeCublasGemm(
        cublas_handle, cublaslt_handle, trans_b, trans_a, n_val, m_val, k_val, b_buffer, ldb, b_data_type, a_buffer,
        lda, a_data_type, c_buffer, ldc, c_data_type, batch_count, alpha, beta, compute_type, stream, workspace_ptr,
        workspace_size, cublaslt_algo));
  };
  return MeasureCudaExecutionTime(cuda_run, stream, warmup_rounds, tested_rounds);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  uint64_t left_bs = 0;
  uint64_t right_bs = 0;
  SetSearchRange(left_bs, right_bs, "batch_size", FLAGS_batch_size, FLAGS_max_batch_size);

  uint64_t left_m = 0;
  uint64_t right_m = 0;
  SetSearchRange(left_m, right_m, "m", FLAGS_m, FLAGS_max_m);

  uint64_t left_n = 0;
  uint64_t right_n = 0;
  SetSearchRange(left_n, right_n, "n", FLAGS_n, FLAGS_max_n);

  uint64_t left_k = 0;
  uint64_t right_k = 0;
  SetSearchRange(left_k, right_k, "k", FLAGS_k, FLAGS_max_k);

  cudaDataType_t a_data_type = SetSearchDataType(FLAGS_input_dtype);
  cudaDataType_t b_data_type = SetSearchDataType(FLAGS_input_dtype);
  cudaDataType_t c_data_type = SetSearchDataType(FLAGS_output_dtype);
  cudaDataType_t compute_type = SetSearchDataType(FLAGS_inner_compute_dtype);

  cublasOperation_t trans_a = SetInputTransop(FLAGS_input_a_transop);
  cublasOperation_t trans_b = SetInputTransop(FLAGS_input_b_transop);

  cudaDeviceProp prop;
  int device_count;
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    throw std::invalid_argument("There is not GPU on you machine.");
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
  uint32_t sm = prop.major * 10 + prop.minor;

  int cuda_ver_tmp;
  CHECK_NVIDIA_CUDA_ERROR(cudaRuntimeGetVersion(&cuda_ver_tmp));
  uint32_t cuda_ver = static_cast<uint32_t>(cuda_ver_tmp);

  std::cout << "search best gemm algo in sm: " << sm << " cuda version: " << cuda_ver << " with batch size range: ["
            << left_bs << ", " << right_bs << "], m range: [" << left_m << ", " << right_m << "], n range: [" << left_n
            << ", " << right_n << "], k range: [" << left_k << ", " << right_k << "]" << std::endl;

  // prepare more buffer for different case
  void* a_buffer;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_buffer, sizeof(float) * right_m * right_k));
  void* b_buffer;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_buffer, sizeof(float) * right_k * right_n));
  void* c_buffer;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&c_buffer, sizeof(float) * right_m * right_n));
  cublasHandle_t cublas_handle;
  CHECK_NVIDIA_CUDA_ERROR(cublasCreate(&cublas_handle));
  cublasLtHandle_t cublaslt_handle;
  CHECK_NVIDIA_CUDA_ERROR(cublasLtCreate(&cublaslt_handle));
  cudaStream_t stream;
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));
  void* cublas_workspace_buffer_ptr;
  size_t workspace_size = llm_kernels::nvidia::GetCublasWorkspaceSize();
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&cublas_workspace_buffer_ptr, workspace_size));

  float alpha = 1.0f;
  float beta = 0.0f;
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
  constexpr int warmup_rounds = 5;
  constexpr int tested_rounds = 5;
  llm_kernels::nvidia::GPUGemmAlgoHelper gpu_gemm_algo_helper;
  if (FLAGS_append_mode) {
    gpu_gemm_algo_helper.LoadFromYaml(FLAGS_output_file);
  }

  // NOTE(karlluo): there is 4 situation in this search process.
  // 1. previous/other library framework GEMM is better, metric is original_time_elapsed_ms
  // 2. cuBlas default GEMM is better, metric is cublas_default_elapsed_ms
  // 3. cuBlas heuristic search algo GEMM is better, metric is min_cublas_h_search_elapsed_ms
  // 4. Ksana custom GEMM is better, metric is min_custom_elapsed_ms(TODO: karlluo)
  for (uint64_t bs_val = left_bs; bs_val <= right_bs; ++bs_val) {
    for (uint64_t m_val = left_m; m_val <= right_m; ++m_val) {
      for (uint64_t n_val = left_n; n_val <= right_n; ++n_val) {
        for (uint64_t k_val = left_k; k_val <= right_k; ++k_val) {
          std::cout << "Searching bs: " << bs_val << ", m: " << m_val << ", n: " << n_val << ", k: " << k_val
                    << std::endl;

          int32_t lda = (trans_a == CUBLAS_OP_N) ? k_val : m_val;
          int32_t ldb = (trans_b == CUBLAS_OP_N) ? n_val : k_val;
          int32_t ldc = n_val;

          // Get previous/other library framework result
          llm_kernels::nvidia::GemmAlgoInfo gemm_algo_info =
              gpu_gemm_algo_helper.GetGemmAlgo(sm, cuda_ver, bs_val, m_val, n_val, k_val, a_data_type, b_data_type,
                                               c_data_type, compute_type, trans_a, trans_b);
          std::cout << "Get previous/other library framework GEMM latency: "
                    << gemm_algo_info.gemm_algo_perf.gpu_elapsed_ms << " ms." << std::endl;
          float original_time_elapsed_ms = gemm_algo_info.gemm_algo_perf.gpu_elapsed_ms;

          // Get default infer latency as baseline
          float cublas_default_elapsed_ms =
              GetCublasGPUElapsedTime(warmup_rounds, tested_rounds, cublas_handle, cublaslt_handle, trans_b, trans_a,
                                      n_val, m_val, k_val, b_buffer, ldb, b_data_type, a_buffer, lda, a_data_type,
                                      c_buffer, ldc, c_data_type, alpha, beta, compute_type, stream, start, stop);

          // Heuristic search DEFAULT_ALGO_SEARCH_NUM GEMM algo candidates
          float min_cublas_h_search_elapsed_ms = cublas_default_elapsed_ms;
          float candidate_algo_time_elapsed_ms = std::numeric_limits<float>::max();
          std::vector<cublasLtMatmulHeuristicResult_t> cublas_algos = llm_kernels::nvidia::HeuristicSearchCublasAlgo(
              cublaslt_handle, trans_b, trans_a, n_val, m_val, k_val, b_buffer, ldb, b_data_type, a_buffer, lda,
              a_data_type, c_buffer, ldc, c_data_type, alpha, beta, compute_type,
              llm_kernels::nvidia::GetCublasWorkspaceSize(), llm_kernels::nvidia::DEFAULT_ALGO_SEARCH_NUM);
          // Loop all GEMM algo candidates and get the lowest latency one as the best algo
          size_t best_algo_idx = 0;
          for (size_t algo_idx = 0; algo_idx < cublas_algos.size(); ++algo_idx) {
            candidate_algo_time_elapsed_ms =
                GetCublasGPUElapsedTime(warmup_rounds, tested_rounds, cublas_handle, cublaslt_handle, trans_b, trans_a,
                                        n_val, m_val, k_val, b_buffer, ldb, b_data_type, a_buffer, lda, a_data_type,
                                        c_buffer, ldc, c_data_type, alpha, beta, compute_type, stream, start, stop,
                                        cublas_workspace_buffer_ptr, workspace_size, &(cublas_algos[algo_idx].algo));

            if (candidate_algo_time_elapsed_ms < min_cublas_h_search_elapsed_ms) {
              min_cublas_h_search_elapsed_ms = candidate_algo_time_elapsed_ms;
              best_algo_idx = algo_idx;
            }
          }

          // TODO(karlluo): support Ksana custom GEMM
          std::vector<float> candidates = {original_time_elapsed_ms, cublas_default_elapsed_ms,
                                           min_cublas_h_search_elapsed_ms};
          std::vector<float>::iterator min_candidates_it = std::min_element(candidates.begin(), candidates.end());
          float min_time_elapsed_ms = *(min_candidates_it);
          if (min_time_elapsed_ms == min_cublas_h_search_elapsed_ms) {
            // Case 1: cuBlas heuristic search algo GEMM is better, metric is min_cublas_h_search_elapsed_ms
            llm_kernels::nvidia::GemmAlgoFingerprint gemm_algo_fingerprint{
                bs_val, m_val, n_val, k_val, trans_a, trans_b, a_data_type, b_data_type, c_data_type, compute_type};
            llm_kernels::nvidia::GemmAlgoInfo algo_info;
            algo_info.gemm_op_type = llm_kernels::nvidia::CUBLASLT_GEMM_ALGO;  // only support cublas from now
            algo_info.cublaslt_algo = cublas_algos[best_algo_idx].algo;
            // TODO(karlluo): record for more performance metrics
            algo_info.gemm_algo_perf.gpu_elapsed_ms = min_time_elapsed_ms;
            // TODO(karlluo): support more GEMM algo
            gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint, algo_info);
            std::cout << "original_time_elapsed_ms: " << original_time_elapsed_ms
                      << ", cublas_default_elapsed_ms: " << cublas_default_elapsed_ms
                      << ", found best algo min_cublas_h_search_elapsed_ms: " << min_cublas_h_search_elapsed_ms
                      << std::endl;
          } else if (min_time_elapsed_ms == cublas_default_elapsed_ms) {
            // Case 2: cuBlas default GEMM is better, metric is cublas_default_elapsed_ms
            llm_kernels::nvidia::GemmAlgoFingerprint gemm_algo_fingerprint{
                bs_val, m_val, n_val, k_val, trans_a, trans_b, a_data_type, b_data_type, c_data_type, compute_type};
            llm_kernels::nvidia::GemmAlgoInfo algo_info;
            algo_info.gemm_op_type = llm_kernels::nvidia::DEFAULT_GEMM_ALGO;  // only support cublas from now
            // TODO(karlluo): record for more performance metrics
            algo_info.gemm_algo_perf.gpu_elapsed_ms = min_time_elapsed_ms;
            // TODO(karlluo): support more GEMM algo
            gpu_gemm_algo_helper.AddGemmAlgo(sm, cuda_ver, gemm_algo_fingerprint, algo_info);
            std::cout << "original_time_elapsed_ms: " << original_time_elapsed_ms
                      << ", min_cublas_h_search_elapsed_ms: " << min_cublas_h_search_elapsed_ms
                      << ", found best algo cublas_default_elapsed_ms: " << cublas_default_elapsed_ms << std::endl;
          } else {
            // Case 3: previous/other library framework GEMM is better, metric is original_time_elapsed_ms
            std::cout << "cublas_default_elapsed_ms: " << cublas_default_elapsed_ms
                      << ", min_cublas_h_search_elapsed_ms: " << min_cublas_h_search_elapsed_ms
                      << ", found best algo original_time_elapsed_ms: " << original_time_elapsed_ms
                      << ", keep original GEMM implement and do not change" << std::endl;
          }
        }
      }
    }
  }
  gpu_gemm_algo_helper.SaveToYaml(FLAGS_output_file);

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_buffer));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_buffer));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(c_buffer));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(cublas_workspace_buffer_ptr));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtDestroy(cublaslt_handle));
  CHECK_NVIDIA_CUDA_ERROR(cublasDestroy(cublas_handle));

  return 0;
}
