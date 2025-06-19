/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <chrono>
#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "permute.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaPermuteTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T>
  void PermuteOnCPU(const T* input, T* output, const std::vector<size_t>& input_shape,
                    const std::vector<size_t>& permutation) {
    std::vector<size_t> output_shape(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
      output_shape[i] = input_shape[permutation[i]];
    }
    std::vector<size_t> input_strides(input_shape.size());
    std::vector<size_t> output_strides(output_shape.size());
    input_strides[input_shape.size() - 1] = 1;
    output_strides[output_shape.size() - 1] = 1;

    for (int i = input_shape.size() - 2; i >= 0; --i) {
      input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
      output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
    size_t total_size = 1;
    for (size_t dim : input_shape) {
      total_size *= dim;
    }
    std::vector<size_t> input_indices(input_shape.size(), 0);
    std::vector<size_t> output_indices(output_shape.size(), 0);
    for (size_t i = 0; i < total_size; ++i) {
      size_t temp = i;
      for (size_t j = 0; j < input_shape.size(); ++j) {
        input_indices[j] = temp / input_strides[j];
        temp %= input_strides[j];
      }
      for (size_t j = 0; j < output_shape.size(); ++j) {
        output_indices[j] = input_indices[permutation[j]];
      }
      size_t output_idx = 0;
      for (size_t j = 0; j < output_shape.size(); ++j) {
        output_idx += output_indices[j] * output_strides[j];
      }
      output[output_idx] = input[i];
    }
  }

  template <typename T, size_t num_dims>
  void TestPermuteAccuracy(const std::vector<size_t>& input_shape, const std::vector<size_t>& permutation) {
    BufferMeta input = CreateBuffer<T>(MemoryType::MEMORY_GPU, input_shape, /*is_random_init*/ true);
    std::vector<size_t> output_shape(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
      output_shape[i] = input_shape[permutation[i]];
    }
    BufferMeta output_gpu = CreateBuffer<T>(MemoryType::MEMORY_GPU, output_shape, /*is_random_init*/ false);
    InvokePermute<num_dims, sizeof(T)>(input.data_ptr, output_gpu.data_ptr, input_shape, permutation, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    BufferMeta input_host = CopyToHost<T>(input);
    BufferMeta output_gpu_host = CopyToHost<T>(output_gpu);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    BufferMeta output_cpu = CreateBuffer<T>(MemoryType::MEMORY_CPU, output_shape, /*is_random_init*/ false);
    PermuteOnCPU<T>(reinterpret_cast<T*>(input_host.data_ptr), reinterpret_cast<T*>(output_cpu.data_ptr), input_shape,
                    permutation);
    bool is_equal = CheckResult<T>("PermuteAccuracyTest", output_gpu_host, output_cpu,
                                   1e-5f,  // atol
                                   1e-8f,  // rtol
                                   0.0f,   // no error tolerance
                                   true    // print detailed results
    );
    EXPECT_TRUE(is_equal) << "GPU and CPU permutation results do not match";
  }

  template <typename T, size_t num_dims>
  void TestPermutePerformance(const std::vector<size_t>& input_shape, const std::vector<size_t>& permutation,
                              int num_iterations = 100) {
    BufferMeta input = CreateBuffer<T>(MemoryType::MEMORY_GPU, input_shape, /*is_random_init*/ true);
    std::vector<size_t> output_shape(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
      output_shape[i] = input_shape[permutation[i]];
    }
    BufferMeta output_gpu = CreateBuffer<T>(MemoryType::MEMORY_GPU, output_shape, /*is_random_init*/ false);
    size_t total_elements = 1;
    for (size_t dim : input_shape) {
      total_elements *= dim;
    }
    for (int i = 0; i < 10; ++i) {
      InvokePermute<num_dims, sizeof(T)>(input.data_ptr, output_gpu.data_ptr, input_shape, permutation, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
      InvokePermute<num_dims, sizeof(T)>(input.data_ptr, output_gpu.data_ptr, input_shape, permutation, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double time_ms = elapsed.count() / num_iterations;
    double throughput = (total_elements * sizeof(T)) / (time_ms * 1e-3) / 1e9;  // GB/s
    std::cout << "Permute Performance (" << num_dims << "D, " << sizeof(T) << " bytes):" << std::endl;
    std::cout << "  Input Shape: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
      std::cout << input_shape[i];
      if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Permutation: [";
    for (size_t i = 0; i < permutation.size(); ++i) {
      std::cout << permutation[i];
      if (i < permutation.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Average Time: " << time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
  }
};

TEST_F(LlamaNvidiaPermuteTestSuit, PermuteAccuracyFloatTest) {
  TestPermuteAccuracy<float, 4>({2, 3, 4, 5},  // input shape
                                {2, 0, 1, 3}   // permutation order
  );
  TestPermuteAccuracy<float, 3>({3, 4, 5},  // input shape
                                {2, 0, 1}   // permutation order
  );
  TestPermuteAccuracy<float, 3>({2, 2, 16},  // input shape
                                {1, 0, 2}    // permutation order
  );
  TestPermuteAccuracy<float, 4>({2, 2, 16, 2},  // input shape
                                {1, 0, 2, 3}    // permutation order
  );
  TestPermuteAccuracy<float, 4>({16, 16, 32, 1},  // input shape
                                {1, 0, 2, 3}   // permutation order
  );
  TestPermuteAccuracy<float, 2>({10, 20},  // input shape
                                {1, 0}     // permutation order
  );
}

// Performance tests
TEST_F(LlamaNvidiaPermuteTestSuit, PermutePerformanceFloatTest) {
  TestPermutePerformance<float, 4>({16, 16, 16, 16},  // input shape
                                   {0, 2, 1, 3}       // permutation order
  );

  TestPermutePerformance<float, 4>({32, 32, 32, 32},  // input shape
                                   {0, 2, 1, 3}       // permutation order
  );

  TestPermutePerformance<float, 4>({32, 32, 32, 32},  // input shape
                                   {1, 0, 2, 3}      // permutation order
  );

  TestPermutePerformance<float, 4>({32, 64, 128, 256},  // input shape
                                   {0, 2, 1, 3}         // permutation order
  );

  // Test typical attention mechanism transpose scenario
  // (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
  TestPermutePerformance<float, 4>({8, 128, 16, 64},  // input shape
                                   {0, 2, 1, 3}       // permutation order
  );
}

TEST_F(LlamaNvidiaPermuteTestSuit, PermuteAccuracyHalfTest) {
  TestPermuteAccuracy<half, 4>({2, 3, 4, 5},  // input shape
                               {2, 0, 1, 3}   // permutation order
  );
  TestPermuteAccuracy<half, 3>({3, 4, 5},  // input shape
                               {2, 0, 1}   // permutation order
  );
  TestPermuteAccuracy<half, 4>({2, 3, 16, 1},  // input shape
                               {1, 0, 2, 3}   // permutation order
  );
  TestPermuteAccuracy<half, 3>({3, 4, 16},  // input shape
                               {1, 0, 2}   // permutation order
  );
  TestPermuteAccuracy<half, 2>({10, 20},  // input shape
                               {1, 0}     // permutation order
  );
}

TEST_F(LlamaNvidiaPermuteTestSuit, PermutePerformanceHalfTest) {
  TestPermutePerformance<half, 4>({16, 16, 16, 16},  // input shape
                                  {0, 2, 1, 3}       // permutation order
  );

  TestPermutePerformance<half, 4>({32, 32, 32, 32},  // input shape
                                  {0, 2, 1, 3}       // permutation order
  );

  TestPermutePerformance<half, 4>({32, 32, 32, 32},  // input shape
                                  {1, 0, 2, 3}       // permutation order
  );

  TestPermutePerformance<half, 4>({32, 32, 32, 1},  // input shape
                                  {1, 0, 2, 3}      // permutation order
  );

  TestPermutePerformance<half, 4>({32, 64, 128, 256},  // input shape
                                  {0, 2, 1, 3}         // permutation order
  );

  TestPermutePerformance<half, 3>({16, 256, 512},  // input shape
                                  {1, 0, 2}        // permutation order
  );

  // Test typical attention mechanism transpose scenario
  // (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
  TestPermutePerformance<half, 4>({8, 128, 16, 64},  // input shape
                                  {0, 2, 1, 3}       // permutation order
  );
}

#ifdef ENABLE_BF16
TEST_F(LlamaNvidiaPermuteTestSuit, PermuteAccuracyBFloat16Test) {
  TestPermuteAccuracy<__nv_bfloat16, 4>({2, 3, 4, 5},  // input shape
                                        {2, 0, 1, 3}   // permutation order
  );
  TestPermuteAccuracy<__nv_bfloat16, 4>({2, 3, 16, 2},  // input shape
                                        {1, 0, 2, 3}    // permutation order
  );
  TestPermuteAccuracy<__nv_bfloat16, 3>({3, 4, 5},  // input shape
                                        {2, 0, 1}   // permutation order
  );
  TestPermuteAccuracy<__nv_bfloat16, 2>({10, 20},  // input shape
                                        {1, 0}     // permutation order
  );
}

TEST_F(LlamaNvidiaPermuteTestSuit, PermutePerformanceBFloat16Test) {
  TestPermutePerformance<__nv_bfloat16, 4>({16, 16, 16, 16},  // input shape
                                           {0, 2, 1, 3}       // permutation order
  );
  TestPermutePerformance<__nv_bfloat16, 4>({16, 16, 16, 16},  // input shape
                                           {1, 0, 2, 3}       // permutation order
  );

  TestPermutePerformance<__nv_bfloat16, 4>({32, 32, 32, 32},  // input shape
                                           {0, 2, 1, 3}       // permutation order
  );

  TestPermutePerformance<__nv_bfloat16, 4>({32, 64, 128, 256},  // input shape
                                           {0, 2, 1, 3}         // permutation order
  );

  TestPermutePerformance<__nv_bfloat16, 3>({128, 256, 512},  // input shape
                                           {1, 0, 2}         // permutation order
  );

  // Test typical attention mechanism transpose scenario
  // (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
  TestPermutePerformance<__nv_bfloat16, 4>({8, 128, 16, 64},  // input shape
                                           {0, 2, 1, 3}       // permutation order
  );
}
#endif

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
