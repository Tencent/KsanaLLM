/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <chrono>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaAsymmetricGemmPreprocessTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    cublasDestroy(cublas_handle);
    cublasLtDestroy(cublaslt_handle);
  }

 protected:
  using NvidiaTestSuitBase::stream;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;

 protected:
  void TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm(const size_t num_rows, const size_t num_cols) {
    int arch = GetSMVersion();
    bool force_interleave = false;
    if (force_interleave && arch == 90) {
      arch = 80;
    }
    QuantType quant_type = QuantType::W4_A16;
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    if (!(details.uses_imma_ldsm)) {
      printf("Pass TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm\n");
      return;
    }

    BufferMeta buffer_permuted_quantized_tensor =
        CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_permuted_quantized_tensor = CopyToDevice<float>(buffer_permuted_quantized_tensor);

    BufferMeta buffer_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, false);
    BufferMeta device_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_rows * num_cols / 2 / 4}, false);

    std::vector<int32_t> row_permutation = get_permutation_map(quant_type);
    BufferMeta device_row_permutation = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {row_permutation.size()}, false);
    cudaMemcpy(device_row_permutation.data_ptr, row_permutation.data(), device_row_permutation.buf_size,
               cudaMemcpyHostToDevice);

    auto cpu_run = [&]() {
      permute_B_rows_for_mixed_gemm(reinterpret_cast<int8_t*>(buffer_quantized_tensor.data_ptr),
                                    reinterpret_cast<const int8_t*>(buffer_permuted_quantized_tensor.data_ptr),
                                    {1, num_rows, num_cols}, quant_type, arch);
    };
    float cpp_time = measureCPUExecutionTime(cpu_run, 1, 2);

    auto cuda_run = [&]() {
      fast_permute_B_rows_for_mixed_gemm(reinterpret_cast<int8_t*>(device_quantized_tensor.data_ptr),
                                         reinterpret_cast<const int8_t*>(device_permuted_quantized_tensor.data_ptr),
                                         reinterpret_cast<const int32_t*>(device_row_permutation.data_ptr),
                                         row_permutation.size(), {1, num_rows, num_cols}, quant_type, arch, stream);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_run, stream);

    printf("permute_B_rows_for_mixed_gemm_kernel(%zu,%zu): cpu time:%f ms, cuda time: %f ms\n", num_rows, num_cols,
           cpp_time, cuda_time);

    EXPECT_TRUE(CheckResult<int32_t>("TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm", buffer_quantized_tensor,
                                     device_quantized_tensor, 1e-5f, 1e-5f));

    DeleteBuffer(buffer_permuted_quantized_tensor);
    DeleteBuffer(buffer_quantized_tensor);
    DeleteBuffer(device_row_permutation);
    DeleteBuffer(device_permuted_quantized_tensor);
    DeleteBuffer(device_quantized_tensor);
  }

  void TestAsymmetricGemmPreprocessSubbyteTranspose(const size_t num_rows, const size_t num_cols) {
    int arch = GetSMVersion();
    bool force_interleave = false;
    if (force_interleave && arch == 90) {
      arch = 80;
    }
    QuantType quant_type = QuantType::W4_A16;
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    if (!(details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR)) {
      printf("Pass TestAsymmetricGemmPreprocessSubbyteTranspose\n");
      return;
    }

    BufferMeta buffer_quantized_tensor =
        CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_quantized_tensor = CopyToDevice<float>(buffer_quantized_tensor);

    BufferMeta buffer_transposed_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, false);
    BufferMeta device_transposed_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_rows * num_cols / 2 / 4}, false);

    auto cpu_run = [&]() {
      subbyte_transpose(reinterpret_cast<int8_t*>(buffer_transposed_quantized_tensor.data_ptr),
                        reinterpret_cast<const int8_t*>(buffer_quantized_tensor.data_ptr), {1, num_rows, num_cols},
                        quant_type);
    };
    float cpp_time = measureCPUExecutionTime(cpu_run, 1, 2);

    auto cuda_run = [&]() {
      fast_subbyte_transpose(reinterpret_cast<int8_t*>(device_transposed_quantized_tensor.data_ptr),
                             reinterpret_cast<const int8_t*>(device_quantized_tensor.data_ptr), {1, num_rows, num_cols},
                             quant_type, stream);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_run, stream);

    printf("subbyte_transpose(%zu,%zu): cpu time:%f ms, cuda time: %f ms\n", num_rows, num_cols, cpp_time, cuda_time);

    EXPECT_TRUE(CheckResult<int32_t>("TestAsymmetricGemmPreprocessSubbyteTranspose", device_transposed_quantized_tensor,
                                     buffer_transposed_quantized_tensor, 1e-5f, 1e-5f));

    DeleteBuffer(buffer_quantized_tensor);
    DeleteBuffer(device_quantized_tensor);
    DeleteBuffer(buffer_transposed_quantized_tensor);
    DeleteBuffer(device_transposed_quantized_tensor);
  }

  void TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor(const size_t num_rows, const size_t num_cols) {
    int arch = GetSMVersion();
    bool force_interleave = false;
    if (force_interleave && arch == 90) {
      arch = 80;
    }
    QuantType quant_type = QuantType::W4_A16;
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    if (!(details.columns_interleaved > 1)) {
      printf("Pass TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor\n");
      return;
    }

    BufferMeta buffer_quantized_tensor =
        CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_quantized_tensor = CopyToDevice<float>(buffer_quantized_tensor);

    BufferMeta buffer_interleaved_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, false);
    BufferMeta device_interleaved_quantized_tensor =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_rows * num_cols / 2 / 4}, false);

    auto cpu_run = [&]() {
      interleave_column_major_tensor(reinterpret_cast<int8_t*>(buffer_interleaved_quantized_tensor.data_ptr),
                                     reinterpret_cast<const int8_t*>(buffer_quantized_tensor.data_ptr),
                                     {1, num_rows, num_cols}, quant_type, details);
    };
    float cpp_time = measureCPUExecutionTime(cpu_run, 1, 2);

    auto cuda_run = [&]() {
      fast_interleave_column_major_tensor(reinterpret_cast<int8_t*>(device_interleaved_quantized_tensor.data_ptr),
                                          reinterpret_cast<const int8_t*>(device_quantized_tensor.data_ptr),
                                          {1, num_rows, num_cols}, quant_type, details, stream);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_run, stream);

    printf("interleave_column_major_tensor(%zu,%zu): cpu time:%f ms, cuda time: %f ms\n", num_rows, num_cols, cpp_time,
           cuda_time);

    EXPECT_TRUE(CheckResult<int32_t>("TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor",
                                     device_interleaved_quantized_tensor, buffer_interleaved_quantized_tensor, 1e-5f,
                                     1e-5f));

    DeleteBuffer(buffer_quantized_tensor);
    DeleteBuffer(device_quantized_tensor);
    DeleteBuffer(buffer_interleaved_quantized_tensor);
    DeleteBuffer(device_interleaved_quantized_tensor);
  }

  void TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace(const size_t num_rows,
                                                                              const size_t num_cols) {
    int arch = GetSMVersion();
    bool force_interleave = false;
    if (force_interleave && arch == 90) {
      arch = 80;
    }

    QuantType quant_type = QuantType::W4_A16;

    size_t num_elts = num_rows * num_cols;

    BufferMeta buffer_tensor = CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_tensor = CopyToDevice<float>(buffer_tensor);

    auto cpu_run = [&]() {
      add_bias_and_interleave_quantized_tensor_inplace(reinterpret_cast<int8_t*>(buffer_tensor.data_ptr), num_elts,
                                                       quant_type);
    };
    float cpp_time = measureCPUExecutionTime(cpu_run, 1, 2);

    auto cuda_run = [&]() {
      fast_add_bias_and_interleave_quantized_tensor_inplace(reinterpret_cast<int8_t*>(device_tensor.data_ptr), num_elts,
                                                            quant_type, stream);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_run, stream);

    printf("add_bias_and_interleave_quantized_tensor_inplace(%zu,%zu): cpu time:%f ms, cuda time: %f ms\n", num_rows,
           num_cols, cpp_time, cuda_time);

    BufferMeta buffer_precision_tensor =
        CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_precision_tensor = CopyToDevice<float>(buffer_precision_tensor);

    add_bias_and_interleave_quantized_tensor_inplace(reinterpret_cast<int8_t*>(buffer_precision_tensor.data_ptr),
                                                     num_elts, quant_type);
    fast_add_bias_and_interleave_quantized_tensor_inplace(reinterpret_cast<int8_t*>(device_precision_tensor.data_ptr),
                                                          num_elts, quant_type, stream);

    EXPECT_TRUE(CheckResult<int32_t>("TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace",
                                     buffer_precision_tensor, device_precision_tensor, 1e-5f, 1e-5f));

    DeleteBuffer(buffer_tensor);
    DeleteBuffer(device_tensor);
    DeleteBuffer(buffer_precision_tensor);
    DeleteBuffer(device_precision_tensor);
  }

  void TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm(const size_t num_rows, const size_t num_cols) {
    QuantType quant_type = QuantType::W4_A16;

    BufferMeta buffer_row_major_quantized_weight =
        CreateBuffer<float>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, true);
    BufferMeta device_row_major_quantized_weight = CopyToDevice<float>(buffer_row_major_quantized_weight);

    BufferMeta buffer_preprocessed_quantized_weight =
        CreateBuffer<int32_t>(MemoryType::MEMORY_CPU, {num_rows * num_cols / 2 / 4}, false);
    BufferMeta device_preprocessed_quantized_weight =
        CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {num_rows * num_cols / 2 / 4}, false);

    auto cpp_begin = std::chrono::high_resolution_clock::now();
    preprocess_weights_for_mixed_gemm(reinterpret_cast<int8_t*>(buffer_preprocessed_quantized_weight.data_ptr),
                                      reinterpret_cast<const int8_t*>(buffer_row_major_quantized_weight.data_ptr),
                                      {1, num_rows, num_cols}, quant_type, false);
    auto cpp_end = std::chrono::high_resolution_clock::now();
    float cpp_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpp_end - cpp_begin).count();

    std::vector<int32_t> row_permutation = get_permutation_map(quant_type);
    BufferMeta device_row_permutation = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {row_permutation.size()}, false);
    cudaMemcpy(device_row_permutation.data_ptr, row_permutation.data(), device_row_permutation.buf_size,
               cudaMemcpyHostToDevice);

    cudaEvent_t cuda_begin, cuda_end;
    cudaEventCreate(&cuda_begin);
    cudaEventCreate(&cuda_end);
    cudaEventRecord(cuda_begin, stream);
    fast_preprocess_weights_for_mixed_gemm(reinterpret_cast<int8_t*>(device_preprocessed_quantized_weight.data_ptr),
                                           reinterpret_cast<int8_t*>(device_row_major_quantized_weight.data_ptr),
                                           reinterpret_cast<const int32_t*>(device_row_permutation.data_ptr),
                                           row_permutation.size(), {1, num_rows, num_cols}, quant_type, false, stream);
    cudaEventRecord(cuda_end, stream);
    cudaEventSynchronize(cuda_end);
    float cuda_time = 0.f;
    cudaEventElapsedTime(&cuda_time, cuda_begin, cuda_end);

    printf("preprocess_weights_for_mixed_gemm(%zu,%zu): cpu time:%f ms, cuda time: %f ms\n", num_rows, num_cols,
           cpp_time, cuda_time);

    EXPECT_TRUE(CheckResult<int32_t>("TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm",
                                     buffer_preprocessed_quantized_weight, device_preprocessed_quantized_weight, 1e-5f,
                                     1e-5f));

    cudaEventDestroy(cuda_begin);
    cudaEventDestroy(cuda_end);

    DeleteBuffer(buffer_row_major_quantized_weight);
    DeleteBuffer(device_row_major_quantized_weight);
    DeleteBuffer(buffer_preprocessed_quantized_weight);
    DeleteBuffer(device_preprocessed_quantized_weight);
    DeleteBuffer(device_row_permutation);
  }
};

TEST_F(NvidiaAsymmetricGemmPreprocessTestSuit, AsymmetricGemmPreprocessTest) {
  TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm(3584, 4608);
  TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm(3584, 3584);
  TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm(3584, 18944);
  TestAsymmetricGemmPreprocessPermuteBRowsForMixedGemm(18944, 3584);

  TestAsymmetricGemmPreprocessSubbyteTranspose(3584, 4608);
  TestAsymmetricGemmPreprocessSubbyteTranspose(3584, 3584);
  TestAsymmetricGemmPreprocessSubbyteTranspose(3584, 18944);
  TestAsymmetricGemmPreprocessSubbyteTranspose(18944, 3584);

  TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor(3584, 4608);
  TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor(3584, 3584);
  TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor(3584, 18944);
  TestAsymmetricGemmPreprocessInterleaveColumnMajorTensor(18944, 3584);

  TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace(3584, 4608);
  TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace(3584, 3584);
  TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace(3584, 18944);
  TestAsymmetricGemmPreprocessAddBiasAndInterleaveQuantizedTensorInplace(18944, 3584);

  TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm(3584, 4608);
  TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm(3584, 3584);
  TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm(3584, 18944);
  TestAsymmetricGemmPreprocessPreprocessWeightsForMixedGemm(18944, 3584);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels