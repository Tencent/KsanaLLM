/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"
#include "identity.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaIdentityTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<size_t, size_t>> test_matrix_sizes = {
      {1, 1},     {1, 4},     {4, 1},     {2, 2},     {3, 3},     {4, 4},     {8, 8},     {16, 16},
      {32, 32},   {64, 64},   {128, 128}, {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096},
      {1, 1024},  {1024, 1},  {32, 64},   {64, 32},   {100, 200}, {200, 100}, {1000, 2000}, {2000, 1000}};

 protected:
  template <typename T>
  void InitIdentityMatrixRef(T* matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        matrix[i * cols + j] = static_cast<T>(i == j ? 1.0f : 0.0f);
      }
    }
  }

  template <typename T>
  void TestInitIdentityMatrix(const size_t rows, const size_t cols) {
    // Create buffers
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {rows, cols}, false);
    BufferMeta output_cpu_meta = CreateBuffer<T>(MemoryType::MEMORY_CPU, {rows, cols}, false);
    BufferMeta ref_cpu_meta = CreateBuffer<T>(MemoryType::MEMORY_CPU, {rows, cols}, false);

    // Initialize reference matrix on CPU
    InitIdentityMatrixRef<T>(reinterpret_cast<T*>(ref_cpu_meta.data_ptr), rows, cols);

    // Run kernel on GPU
    InitIdentityMatrixAdaptive<T>(reinterpret_cast<T*>(output_meta.data_ptr), rows, cols, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy result to CPU for comparison
    output_cpu_meta = CopyToHost<T>(output_meta);

    // Verify correctness
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    } else if (std::is_same<T, uint8_t>::value) {
      type_str = "uint8";
    }

    EXPECT_TRUE(CheckResult<T>("identity_matrix_" + type_str + "_rows_" + std::to_string(rows) + "_cols_" +
                               std::to_string(cols), ref_cpu_meta, output_cpu_meta, 1e-5f, 1e-5f));

    // Performance test for larger matrices
    if (rows * cols >= 1024) {
      const int warmup_times = 5;
      const int test_times = 10;
      auto cuda_run = [&]() {
        InitIdentityMatrixAdaptive<T>(reinterpret_cast<T*>(output_meta.data_ptr), rows, cols, stream);
      };
      float time_elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmup_times, test_times);

      std::cout << "identity_matrix_" << type_str << "_rows_" << rows << "_cols_" << cols
                << " time elapsed: " << time_elapsed_ms << " ms" << std::endl;
    }

    // Cleanup
    DeleteBuffer(output_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(ref_cpu_meta);
  }
};

TEST_F(LlamaNvidiaIdentityTestSuit, FloatIdentityMatrixTest) {
  for (const auto& size_pair : test_matrix_sizes) {
    TestInitIdentityMatrix<float>(size_pair.first, size_pair.second);
  }
}

TEST_F(LlamaNvidiaIdentityTestSuit, HalfIdentityMatrixTest) {
  for (const auto& size_pair : test_matrix_sizes) {
    TestInitIdentityMatrix<half>(size_pair.first, size_pair.second);
  }
}

TEST_F(LlamaNvidiaIdentityTestSuit, BFloat16IdentityMatrixTest) {
  for (const auto& size_pair : test_matrix_sizes) {
    TestInitIdentityMatrix<__nv_bfloat16>(size_pair.first, size_pair.second);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
