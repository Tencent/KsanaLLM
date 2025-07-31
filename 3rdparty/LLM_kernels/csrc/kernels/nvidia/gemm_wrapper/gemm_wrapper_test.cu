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

struct LlamaNvidiaGemmCublasTestOpPair {
  cublasOperation_t transa;
  cublasOperation_t transb;
};

class LlamaNvidiaGemmWrapperTestSuit : public NvidiaTestSuitBase {
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
  const std::vector<LlamaNvidiaGemmCublasTestOpPair> cublas_op_pairs{
      {CUBLAS_OP_N, CUBLAS_OP_N}, {CUBLAS_OP_N, CUBLAS_OP_T}, {CUBLAS_OP_T, CUBLAS_OP_N}, {CUBLAS_OP_T, CUBLAS_OP_T}};

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void PrepareComputeType(cudaDataType_t& atype, cudaDataType_t& btype, cudaDataType_t& ctype,
                          cudaDataType_t& compute_type) {
    if (std::is_same<INPUT_DTYPE, float>::value) {
      atype = CUDA_R_32F;
      btype = CUDA_R_32F;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<INPUT_DTYPE, half>::value) {
      atype = CUDA_R_16F;
      btype = CUDA_R_16F;
      ctype = CUDA_R_16F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<INPUT_DTYPE, __nv_bfloat16>::value) {
      atype = CUDA_R_16BF;
      btype = CUDA_R_16BF;
      ctype = CUDA_R_16BF;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<INPUT_DTYPE, __nv_fp8_e4m3>::value) {
      atype = CUDA_R_8F_E4M3;
      btype = CUDA_R_8F_E4M3;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else {
      throw std::runtime_error(
          "Unknown test type in ComputeReference. Only support float, float16, __nv_bfloat16 and __nv_fp8_e4m3.");
    }
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void ComputeReference(const cublasOperation_t transa, const cublasOperation_t transb, const void* a_ptr,
                        const void* b_ptr, void* c_ptr, size_t m, size_t n, size_t k, float alpha = 1.0f,
                        float beta = 0.0f) {
    size_t lda = (transa == CUBLAS_OP_N) ? k : m;
    size_t ldb = (transb == CUBLAS_OP_N) ? n : k;
    size_t ldc = n;

    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    PrepareComputeType<INPUT_DTYPE, OUTPUT_DTYPE>(atype, btype, ctype, compute_type);

    CHECK_NVIDIA_CUDA_ERROR(cublasGemmEx(cublas_handle, transb, transa, n, m, k, (const void*)&alpha, b_ptr, btype, ldb,
                                         a_ptr, atype, lda, (const void*)&beta, c_ptr, ctype, ldc, compute_type,
                                         CUBLAS_GEMM_DEFAULT));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
  }

  template <typename T>
  std::string GenerateCublasGemmTestName(const std::string& task_name, const cublasOperation_t transa,
                                         const cublasOperation_t transb, const size_t batch_size, const size_t m,
                                         const size_t n, const size_t k) {
    std::string result = task_name;
    result += "_b_" + std::to_string(batch_size);
    result += "_m_" + std::to_string(m);
    result += "_n_" + std::to_string(n);
    result += "_k_" + std::to_string(k);

    if (std::is_same<T, float>::value) {
      result += "_float";
    } else if (std::is_same<T, half>::value) {
      result += "_half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      result += "_bfloat16";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      result += "_fp8_e4m3";
    } else {
      throw std::runtime_error(
          "Unknown test type in GenerateCublasGemmTestName. Only support float, float16, __nv_bfloat16 and "
          "__nv_fp8_e4m3.");
    }

    result += (transa == CUBLAS_OP_N) ? "_AN_" : "_AT_";
    result += (transb == CUBLAS_OP_N) ? "BN" : "BT";

    return result;
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void TestCublasBatch(size_t batch_size, size_t m, size_t n, size_t k) {
    BufferMeta a_buffer =
        CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer =
        CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, k, n}, /*is_random_init*/ true);
    BufferMeta c_buffer =
        CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer =
        CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {batch_size, m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    PrepareComputeType<INPUT_DTYPE, OUTPUT_DTYPE>(atype, btype, ctype, compute_type);

    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      int64_t stridea = m * k;
      int64_t strideb = k * n;
      int64_t stridec = m * n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<INPUT_DTYPE>(std::string("TestCublasBatch"), op_pair.transa,
                                                                      op_pair.transb, batch_size, m, n, k);

      for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        ComputeReference<INPUT_DTYPE, OUTPUT_DTYPE>(
            op_pair.transa, op_pair.transb, (const void*)(((INPUT_DTYPE*)a_buffer.data_ptr) + stridea * batch_idx),
            (const void*)(((INPUT_DTYPE*)b_buffer.data_ptr) + strideb * batch_idx),
            (void*)(((OUTPUT_DTYPE*)expected_buffer.data_ptr) + stridec * batch_idx), m, n, k);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasStridedBatchedGemm(
          cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, strideb,
          btype, a_buffer.data_ptr, lda, stridea, atype, c_buffer.data_ptr, ldc, stridec, ctype, batch_size,
          compute_type, alpha, beta));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_invokeCublasStridedBatchedGemm_1", expected_buffer, c_buffer,
                                            1e-4f, 1e-5f, true));
    }

    DeleteBuffer(expected_buffer);
    DeleteBuffer(c_buffer);
    DeleteBuffer(b_buffer);
    DeleteBuffer(a_buffer);
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void TestCublasGemm(size_t m, size_t n, size_t k) {
    BufferMeta a_buffer = CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer = CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {k, n}, /*is_random_init*/ true);
    BufferMeta c_buffer = CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer = CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    PrepareComputeType<INPUT_DTYPE, OUTPUT_DTYPE>(atype, btype, ctype, compute_type);
    int batch_size = 1;
    size_t default_ws_size = GetCublasWorkspaceSize();

    // test correctness
    float miss_match_rate = 0.01f;
    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<INPUT_DTYPE>(std::string("TestCublasGemmCorrectnessMatmul"),
                                                                      op_pair.transa, op_pair.transb, 1ul, m, n, k);
      // compute the reference
      ComputeReference<INPUT_DTYPE, OUTPUT_DTYPE>(op_pair.transa, op_pair.transb, (const void*)a_buffer.data_ptr,
                                                  (const void*)b_buffer.data_ptr, expected_buffer.data_ptr, m, n, k);
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemmEx(cublas_handle, op_pair.transb, op_pair.transa, n, m, k,
                                                 (const void*)&alpha, b_buffer.data_ptr, btype, ldb, a_buffer.data_ptr,
                                                 atype, lda, (const void*)&beta, c_buffer.data_ptr, ctype, ldc,
                                                 compute_type, CUBLAS_GEMM_DEFAULT));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_invokeCublasGemmEx", expected_buffer, c_buffer, 1e-4f, 1e-5f,
                                            miss_match_rate, true));

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_invokeCublasGemm_1", expected_buffer, c_buffer, 1e-4f, 1e-5f,
                                            miss_match_rate, true));

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, batch_size, alpha, beta, compute_type,
                                               stream, nullptr, 0, nullptr));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_invokeCublasGemm_2", expected_buffer, c_buffer, 1e-4f, 1e-5f,
                                            miss_match_rate, true));

      cublasLtMatmulAlgo_t cublaslt_algo = HeuristicSearchCublasAlgo(
          cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr,
          lda, atype, c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, GetCublasWorkspaceSize());
      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, batch_size, alpha, beta, compute_type,
                                               stream, cublas_workspace_buffer_ptr, default_ws_size, &cublaslt_algo));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_invokeCublasGemm_3", expected_buffer, c_buffer, 1e-4f, 1e-5f,
                                            miss_match_rate, true));
    }

    // test performance
    constexpr int warmup_rounds = 5;
    constexpr int tested_rounds = 5;
    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<INPUT_DTYPE>(std::string("TestCublasGemmPerformance"),
                                                                      op_pair.transa, op_pair.transb, 1ul, m, n, k);

      auto InvokeCublasGemmEx_cuda_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemmEx(cublas_handle, op_pair.transb, op_pair.transa, n, m, k,
                                                   (const void*)&alpha, b_buffer.data_ptr, btype, ldb,
                                                   a_buffer.data_ptr, atype, lda, (const void*)&beta, c_buffer.data_ptr,
                                                   ctype, ldc, compute_type, CUBLAS_GEMM_DEFAULT));
      };
      float InvokeCublasGemmEx_time_elapsed_ms =
          MeasureCudaExecutionTime(InvokeCublasGemmEx_cuda_run, stream, warmup_rounds, tested_rounds);

      auto InvokeCublasGemm_1_cuda_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      };
      float InvokeCublasGemm_1_time_elapsed_ms =
          MeasureCudaExecutionTime(InvokeCublasGemm_1_cuda_run, stream, warmup_rounds, tested_rounds);

      auto InvokeCublasGemm_2_cuda_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, batch_size, alpha, beta, compute_type,
                                                 stream, nullptr, 0, nullptr));
      };
      float InvokeCublasGemm_2_time_elapsed_ms =
          MeasureCudaExecutionTime(InvokeCublasGemm_2_cuda_run, stream, warmup_rounds, tested_rounds);

      cublasLtMatmulAlgo_t cublaslt_algo = HeuristicSearchCublasAlgo(
          cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr,
          lda, atype, c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, GetCublasWorkspaceSize());
      auto InvokeCublasGemm_3_cuda_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, batch_size, alpha, beta, compute_type,
                                                 stream, cublas_workspace_buffer_ptr, default_ws_size, &cublaslt_algo));
      };
      float InvokeCublasGemm_3_time_elapsed_ms =
          MeasureCudaExecutionTime(InvokeCublasGemm_3_cuda_run, stream, warmup_rounds, tested_rounds);

      std::cout << test_name << " performance InvokeCublasGemmEx: " << InvokeCublasGemmEx_time_elapsed_ms
                << ", InvokeCublasGemm_1: " << InvokeCublasGemm_1_time_elapsed_ms
                << ", InvokeCublasGemm_2: " << InvokeCublasGemm_2_time_elapsed_ms
                << ", InvokeCublasGemm_3(with heuristic search gemm algo): " << InvokeCublasGemm_3_time_elapsed_ms
                << std::endl;
    }

    DeleteBuffer(expected_buffer);
    DeleteBuffer(c_buffer);
    DeleteBuffer(b_buffer);
    DeleteBuffer(a_buffer);
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void CublasGemmTest() {
    std::vector<size_t> batch_sizes = {1, 2, 7};

    using testcase_t = std::tuple<size_t, size_t, size_t>;
    std::vector<testcase_t> testcases = {{16, 32, 64}, {255, 255, 255}, {1041, 999, 1}};

    for (testcase_t& tc : testcases) {
      size_t m = std::get<0>(tc);
      size_t n = std::get<1>(tc);
      size_t k = std::get<2>(tc);
      TestCublasGemm<INPUT_DTYPE, OUTPUT_DTYPE>(m, n, k);
    }

    for (size_t bs : batch_sizes) {
      for (testcase_t& tc : testcases) {
        size_t m = std::get<0>(tc);
        size_t n = std::get<1>(tc);
        size_t k = std::get<2>(tc);
        TestCublasBatch<INPUT_DTYPE, OUTPUT_DTYPE>(bs, m, n, k);
      }
    }
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void TestCustomGemm(size_t m, size_t n, size_t k) {
    BufferMeta a_buffer = CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer = CreateBuffer<INPUT_DTYPE>(MemoryType::MEMORY_GPU, {n, k}, /*is_random_init*/ true);
    BufferMeta c_buffer = CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer = CreateBuffer<OUTPUT_DTYPE>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    PrepareComputeType<INPUT_DTYPE, OUTPUT_DTYPE>(atype, btype, ctype, compute_type);

    // test correctness
    float miss_match_rate = 0.01f;
    for (auto& op_pair : cublas_op_pairs) {
      if (op_pair.transa != CUBLAS_OP_N || op_pair.transb != CUBLAS_OP_T) {
        continue;
      }
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;

      std::string test_name = GenerateCublasGemmTestName<INPUT_DTYPE>(std::string("TestCustomGemmCorrectnessMatmul"),
                                                                      op_pair.transa, op_pair.transb, 1ul, m, n, k);
      // compute the reference
      ComputeReference<INPUT_DTYPE, OUTPUT_DTYPE>(op_pair.transa, op_pair.transb, (const void*)a_buffer.data_ptr,
                                                  (const void*)b_buffer.data_ptr, expected_buffer.data_ptr, m, n, k);
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_NVIDIA_CUDA_ERROR(InvokeCustomGemm(stream, op_pair.transa, op_pair.transb, m, n, k, a_buffer.data_ptr, lda,
                                               atype, b_buffer.data_ptr, ldb, btype, c_buffer.data_ptr, ldc, ctype,
                                               compute_type, alpha));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(CheckResult<OUTPUT_DTYPE>(test_name + "_InvokeCustomGemm", c_buffer, expected_buffer, 1e-3f, 1e-3f,
                                            miss_match_rate, true));
    }

    // test performance
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
    constexpr int warmup_rounds = 5;
    constexpr int tested_rounds = 5;
    for (auto& op_pair : cublas_op_pairs) {
      if (op_pair.transa != CUBLAS_OP_N || op_pair.transb != CUBLAS_OP_T) {
        continue;
      }
      std::string test_name = GenerateCublasGemmTestName<INPUT_DTYPE>(std::string("TestCustomGemmPerformance"),
                                                                      op_pair.transa, op_pair.transb, 1ul, m, n, k);
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;
      float beta = 0.0f;
      int batch_size = 1;
      size_t default_ws_size = GetCublasWorkspaceSize();

      // original cublas
      auto original_cublas_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      };
      float cublas_time_elapsed_ms =
          MeasureCudaExecutionTime(original_cublas_run, stream, warmup_rounds, tested_rounds);

      // cublas with heuristic Search
      float candidate_algo_time_elapsed_ms = std::numeric_limits<float>::max();
      float min_time_elapsed_ms = cublas_time_elapsed_ms;
      std::vector<cublasLtMatmulHeuristicResult_t> cublas_algos = HeuristicSearchCublasAlgo(
          cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr,
          lda, atype, c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, GetCublasWorkspaceSize(),
          /*top_algo_num*/ DEFAULT_ALGO_SEARCH_NUM);

      for (size_t algo_idx = 0; algo_idx < cublas_algos.size(); ++algo_idx) {
        auto cuda_run = [&]() {
          CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(
              cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype,
              a_buffer.data_ptr, lda, atype, c_buffer.data_ptr, ldc, ctype, batch_size, alpha, beta, compute_type,
              stream, cublas_workspace_buffer_ptr, default_ws_size, &(cublas_algos[algo_idx].algo)));
        };
        candidate_algo_time_elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmup_rounds, tested_rounds);

        if (candidate_algo_time_elapsed_ms < min_time_elapsed_ms) {
          min_time_elapsed_ms = candidate_algo_time_elapsed_ms;
        }
      }

      auto custom_with_compute_fp32_dtype_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCustomGemm(stream, op_pair.transa, op_pair.transb, m, n, k, a_buffer.data_ptr,
                                                 lda, atype, b_buffer.data_ptr, ldb, btype, c_buffer.data_ptr, ldc,
                                                 ctype, compute_type, alpha));
      };
      float custom_with_compute_fp32_dtype_time_elapsed_ms =
          MeasureCudaExecutionTime(custom_with_compute_fp32_dtype_run, stream, warmup_rounds, tested_rounds);

      auto custom_gemm_run = [&]() {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCustomGemm(stream, op_pair.transa, op_pair.transb, m, n, k, a_buffer.data_ptr,
                                                 lda, atype, b_buffer.data_ptr, ldb, btype, c_buffer.data_ptr, ldc,
                                                 ctype, compute_type, alpha));
      };
      float custom_gemm_time_elapsed_ms =
          MeasureCudaExecutionTime(custom_gemm_run, stream, warmup_rounds, tested_rounds);

      // custom gemm must better than cublas
      EXPECT_GT(min_time_elapsed_ms, custom_with_compute_fp32_dtype_time_elapsed_ms);

      std::cout << "m: " << m << ", n: " << n << ", k: " << k << ", finally performnance enhance: "
                << (min_time_elapsed_ms - custom_with_compute_fp32_dtype_time_elapsed_ms) / min_time_elapsed_ms * 100.0
                << "%"
                << ", cublas_time_elapsed_ms: " << cublas_time_elapsed_ms
                << ", heuristic_search_time_elapsed_ms: " << min_time_elapsed_ms
                << ", custom_with_compute_fp32_dtype_time_elapsed_ms: "
                << custom_with_compute_fp32_dtype_time_elapsed_ms
                << ", custom_gemm_time_elapsed_ms: " << custom_gemm_time_elapsed_ms << std::endl;
    }

    DeleteBuffer(expected_buffer);
    DeleteBuffer(c_buffer);
    DeleteBuffer(b_buffer);
    DeleteBuffer(a_buffer);
  }

  template <typename INPUT_DTYPE, typename OUTPUT_DTYPE>
  void CustomGemmTest() {
    using testcase_t = std::tuple<size_t, size_t, size_t>;
    std::vector<testcase_t> testcases = {{1, 6912, 5120}, {2, 6912, 5120}, {3, 6912, 5120},
                                         {4, 6912, 5120}, {4, 7680, 5120}, {4, 5120, 2560}};

    for (testcase_t& tc : testcases) {
      size_t m = std::get<0>(tc);
      size_t n = std::get<1>(tc);
      size_t k = std::get<2>(tc);
      TestCustomGemm<INPUT_DTYPE, OUTPUT_DTYPE>(m, n, k);
    }
  }
};

TEST_F(LlamaNvidiaGemmWrapperTestSuit, CublasGemmTest) {
  CublasGemmTest<float, float>();
  CublasGemmTest<half, half>();
  CublasGemmTest<__nv_bfloat16, __nv_bfloat16>();
}

TEST_F(LlamaNvidiaGemmWrapperTestSuit, CustomGemmTest) {
  CustomGemmTest<float, float>();
  CustomGemmTest<half, half>();
  CustomGemmTest<__nv_bfloat16, __nv_bfloat16>();
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
