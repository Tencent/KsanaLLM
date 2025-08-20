/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/add_mul/add_mul.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaAddMulTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  const std::vector<std::pair<int, int>> m_n_pairs_float = {
      {2048, 7168},
      {4, 256},
      {1, 1}
  };
  const std::vector<std::pair<int, int>> m_n_pairs_half = {
      {60, 7168},
      {2, 128},
      {3, 1}
  };
  const std::vector<std::pair<int, int>> m_n_pairs_bf16 = {
      {30, 7168},
      {128, 1},
      {1, 4}
  };

 protected:
  // CPU reference implementations for different add-mul operations

  // Reference for output = (input1 + input2) * scale
  void AddThenMulFloatRef(const size_t m, const size_t n, const float* input1, const float* input2,
                          const float scale, float* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = (input1[i] + input2[i]) * scale;
    }
  }

  void AddThenMulHalfRef(const size_t m, const size_t n, const half* input1, const half* input2,
                         const half scale, half* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = (half)((((half_float::half)input1[i]) + ((half_float::half)input2[i])) * (half_float::half)scale);
    }
  }

  void AddThenMulBf16Ref(const size_t m, const size_t n, const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                         const __nv_bfloat16 scale, __nv_bfloat16* output) {
    for (size_t i = 0; i < m * n; ++i) {
      float a = CastBFloat16ToFloat32(static_cast<uint16_t>(input1[i]));
      float b = CastBFloat16ToFloat32(static_cast<uint16_t>(input2[i]));
      float s = CastBFloat16ToFloat32(static_cast<uint16_t>(scale));
      output[i] = __float2bfloat16((a + b) * s);
    }
  }

  // Reference for output = input1 + input2 * scale
  void AddMulSecondFloatRef(const size_t m, const size_t n, const float* input1, const float* input2,
                            const float scale, float* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = input1[i] + input2[i] * scale;
    }
  }

  void AddMulSecondHalfRef(const size_t m, const size_t n, const half* input1, const half* input2,
                           const half scale, half* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = (half)(((half_float::half)input1[i]) + ((half_float::half)input2[i]) * (half_float::half)scale);
    }
  }

  void AddMulSecondBf16Ref(const size_t m, const size_t n, const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                           const __nv_bfloat16 scale, __nv_bfloat16* output) {
    for (size_t i = 0; i < m * n; ++i) {
      float a = CastBFloat16ToFloat32(static_cast<uint16_t>(input1[i]));
      float b = CastBFloat16ToFloat32(static_cast<uint16_t>(input2[i]));
      float s = CastBFloat16ToFloat32(static_cast<uint16_t>(scale));
      output[i] = __float2bfloat16(a + b * s);
    }
  }

  // Reference for output = (input1 + input2 + bias) * scale
  void AddBiasThenMulFloatRef(const size_t m, const size_t n, const float* input1, const float* input2,
                              const float* bias, const float scale, float* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        float bias_value = (bias == nullptr) ? 0.0f : bias[n_idx];
        output[m_idx * n + n_idx] = (input1[m_idx * n + n_idx] + input2[m_idx * n + n_idx] + bias_value) * scale;
      }
    }
  }

  void AddBiasThenMulHalfRef(const size_t m, const size_t n, const half* input1, const half* input2,
                             const half* bias, const half scale, half* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        half bias_value = (bias == nullptr) ? static_cast<half>(0.0f) : bias[n_idx];
        output[m_idx * n + n_idx] = (half)((((half_float::half)input1[m_idx * n + n_idx]) +
                                           ((half_float::half)input2[m_idx * n + n_idx]) +
                                           (half_float::half)bias_value) * (half_float::half)scale);
      }
    }
  }

  void AddBiasThenMulBf16Ref(const size_t m, const size_t n, const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                             const __nv_bfloat16* bias, const __nv_bfloat16 scale, __nv_bfloat16* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        float bias_value = (bias == nullptr) ? 0.0f : CastBFloat16ToFloat32(static_cast<uint16_t>(bias[n_idx]));
        float a = CastBFloat16ToFloat32(static_cast<uint16_t>(input1[m_idx * n + n_idx]));
        float b = CastBFloat16ToFloat32(static_cast<uint16_t>(input2[m_idx * n + n_idx]));
        float s = CastBFloat16ToFloat32(static_cast<uint16_t>(scale));
        output[m_idx * n + n_idx] = __float2bfloat16((a + b + bias_value) * s);
      }
    }
  }

  // Reference for output = input1 * scale1 + input2 * scale2
  void MulThenAddFloatRef(const size_t m, const size_t n, const float* input1, const float* input2,
                          const float scale1, const float scale2, float* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = input1[i] * scale1 + input2[i] * scale2;
    }
  }

  void MulThenAddHalfRef(const size_t m, const size_t n, const half* input1, const half* input2,
                         const half scale1, const half scale2, half* output) {
    for (size_t i = 0; i < m * n; ++i) {
      output[i] = (half)(((half_float::half)input1[i]) * ((half_float::half)scale1) +
                         ((half_float::half)input2[i]) * ((half_float::half)scale2));
    }
  }

  void MulThenAddBf16Ref(const size_t m, const size_t n, const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                         const __nv_bfloat16 scale1, const __nv_bfloat16 scale2, __nv_bfloat16* output) {
    for (size_t i = 0; i < m * n; ++i) {
      float a = CastBFloat16ToFloat32(static_cast<uint16_t>(input1[i]));
      float b = CastBFloat16ToFloat32(static_cast<uint16_t>(input2[i]));
      float s1 = CastBFloat16ToFloat32(static_cast<uint16_t>(scale1));
      float s2 = CastBFloat16ToFloat32(static_cast<uint16_t>(scale2));
      output[i] = __float2bfloat16(a * s1 + b * s2);
    }
  }

  // Template test function for InvokeAddThenMul
  template <typename T>
  void TestAddThenMul(const size_t m, const size_t n) {
    // Create input and output buffers
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    BufferMeta input1_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ true);
    BufferMeta input2_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ true);
    
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input1_cpu_meta = CopyToHost<T>(input1_meta);
    BufferMeta input2_cpu_meta = CopyToHost<T>(input2_meta);

    T scale = static_cast<T>(0.5f);
    std::string type_str;

    // Compute reference result
    if (std::is_same<T, float>::value) {
      type_str = "float";
      AddThenMulFloatRef(m, n, reinterpret_cast<const float*>(input1_cpu_meta.data_ptr),
                         reinterpret_cast<const float*>(input2_cpu_meta.data_ptr),
                         *reinterpret_cast<const float*>(&scale),
                         reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, half>::value) {
      type_str = "half";
      AddThenMulHalfRef(m, n, reinterpret_cast<const half*>(input1_cpu_meta.data_ptr),
                        reinterpret_cast<const half*>(input2_cpu_meta.data_ptr),
                        *reinterpret_cast<const half*>(&scale),
                        reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      AddThenMulBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input1_cpu_meta.data_ptr),
                        reinterpret_cast<const __nv_bfloat16*>(input2_cpu_meta.data_ptr),
                        *reinterpret_cast<const __nv_bfloat16*>(&scale),
                        reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    }

    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // Call GPU kernel
    InvokeAddThenMul<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                        reinterpret_cast<const T*>(input1_meta.data_ptr),
                        reinterpret_cast<const T*>(input2_meta.data_ptr),
                        scale, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // Check result with appropriate tolerance
    float abs_tol = 1e-5f, rel_tol = 1e-5f;
    if (std::is_same<T, half>::value) {
      abs_tol = 1e-3f; rel_tol = 1e-3f;  // Relaxed tolerance for half precision
    }
    EXPECT_TRUE(CheckResult<T>("add_then_mul_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, abs_tol, rel_tol));

    // Performance test
    const int kWarmUpRounds = 5;
    const int kRounds = 10;

    // Warmup
    for (int i = 0; i < kWarmUpRounds; i++) {
      InvokeAddThenMul<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                          reinterpret_cast<const T*>(input1_meta.data_ptr),
                          reinterpret_cast<const T*>(input2_meta.data_ptr),
                          scale, m, n, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Performance measurement
    cudaEvent_t start, stop;
    float time_elapsed_ms = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));

    for (int i = 0; i < kRounds; i++) {
      InvokeAddThenMul<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                          reinterpret_cast<const T*>(input1_meta.data_ptr),
                          reinterpret_cast<const T*>(input2_meta.data_ptr),
                          scale, m, n, stream);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&time_elapsed_ms, start, stop));

    std::cout << "add_then_mul_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n)
              << " time elapsed : " << time_elapsed_ms / kRounds << " ms" << std::endl;

    // Cleanup
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop));
    DeleteBuffer(output_ref_meta);
    DeleteBuffer(input2_cpu_meta);
    DeleteBuffer(input1_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input1_meta);
    DeleteBuffer(input2_meta);
  }

  // Template test function for InvokeAddMulSecond
  template <typename T>
  void TestAddMulSecond(const size_t m, const size_t n) {
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, false);
    BufferMeta input1_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    BufferMeta input2_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input1_cpu_meta = CopyToHost<T>(input1_meta);
    BufferMeta input2_cpu_meta = CopyToHost<T>(input2_meta);

    T scale = static_cast<T>(1.5f);
    std::string type_str;

    if (std::is_same<T, float>::value) {
      type_str = "float";
      AddMulSecondFloatRef(m, n, reinterpret_cast<const float*>(input1_cpu_meta.data_ptr),
                           reinterpret_cast<const float*>(input2_cpu_meta.data_ptr),
                           *reinterpret_cast<const float*>(&scale),
                           reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, half>::value) {
      type_str = "half";
      AddMulSecondHalfRef(m, n, reinterpret_cast<const half*>(input1_cpu_meta.data_ptr),
                          reinterpret_cast<const half*>(input2_cpu_meta.data_ptr),
                          *reinterpret_cast<const half*>(&scale),
                          reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      AddMulSecondBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input1_cpu_meta.data_ptr),
                          reinterpret_cast<const __nv_bfloat16*>(input2_cpu_meta.data_ptr),
                          *reinterpret_cast<const __nv_bfloat16*>(&scale),
                          reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    }

    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    InvokeAddMulSecond<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                          reinterpret_cast<const T*>(input1_meta.data_ptr),
                          reinterpret_cast<const T*>(input2_meta.data_ptr),
                          scale, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Use different tolerance for different data types
    float abs_tol = 1e-5f, rel_tol = 1e-5f;
    if (std::is_same<T, half>::value) {
      abs_tol = 1e-3f; rel_tol = 1e-3f;  // Relaxed tolerance for half precision
    }
    EXPECT_TRUE(CheckResult<T>("add_mul_second_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, abs_tol, rel_tol));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(input2_cpu_meta);
    DeleteBuffer(input1_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input1_meta);
    DeleteBuffer(input2_meta);
  }

  // Template test function for InvokeAddBiasThenMul
  template <typename T>
  void TestAddBiasThenMul(const size_t m, const size_t n) {
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, false);
    BufferMeta input1_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    BufferMeta input2_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    BufferMeta bias_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n}, true);
    
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input1_cpu_meta = CopyToHost<T>(input1_meta);
    BufferMeta input2_cpu_meta = CopyToHost<T>(input2_meta);
    BufferMeta bias_cpu_meta = CopyToHost<T>(bias_meta);

    T scale = static_cast<T>(0.8f);
    std::string type_str;

    if (std::is_same<T, float>::value) {
      type_str = "float";
      AddBiasThenMulFloatRef(m, n, reinterpret_cast<const float*>(input1_cpu_meta.data_ptr),
                             reinterpret_cast<const float*>(input2_cpu_meta.data_ptr),
                             reinterpret_cast<const float*>(bias_cpu_meta.data_ptr),
                             *reinterpret_cast<const float*>(&scale),
                             reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, half>::value) {
      type_str = "half";
      AddBiasThenMulHalfRef(m, n, reinterpret_cast<const half*>(input1_cpu_meta.data_ptr),
                            reinterpret_cast<const half*>(input2_cpu_meta.data_ptr),
                            reinterpret_cast<const half*>(bias_cpu_meta.data_ptr),
                            *reinterpret_cast<const half*>(&scale),
                            reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      AddBiasThenMulBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input1_cpu_meta.data_ptr),
                            reinterpret_cast<const __nv_bfloat16*>(input2_cpu_meta.data_ptr),
                            reinterpret_cast<const __nv_bfloat16*>(bias_cpu_meta.data_ptr),
                            *reinterpret_cast<const __nv_bfloat16*>(&scale),
                            reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    }

    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    InvokeAddBiasThenMul<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                            reinterpret_cast<const T*>(input1_meta.data_ptr),
                            reinterpret_cast<const T*>(input2_meta.data_ptr),
                            reinterpret_cast<const T*>(bias_meta.data_ptr),
                            scale, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Use appropriate tolerance for different data types
    float abs_tol = 1e-5f, rel_tol = 1e-5f;
    if (std::is_same<T, half>::value) {
      abs_tol = 1e-3f; rel_tol = 1e-3f;  // Relaxed tolerance for half precision
    }
    EXPECT_TRUE(CheckResult<T>("add_bias_then_mul_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, abs_tol, rel_tol));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(bias_cpu_meta);
    DeleteBuffer(input2_cpu_meta);
    DeleteBuffer(input1_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input1_meta);
    DeleteBuffer(input2_meta);
    DeleteBuffer(bias_meta);
  }

  // Template test function for InvokeMulThenAdd
  template <typename T>
  void TestMulThenAdd(const size_t m, const size_t n) {
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, false);
    BufferMeta input1_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    BufferMeta input2_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, true);
    
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input1_cpu_meta = CopyToHost<T>(input1_meta);
    BufferMeta input2_cpu_meta = CopyToHost<T>(input2_meta);

    T scale1 = static_cast<T>(0.3f);
    T scale2 = static_cast<T>(0.7f);
    std::string type_str;

    if (std::is_same<T, float>::value) {
      type_str = "float";
      MulThenAddFloatRef(m, n, reinterpret_cast<const float*>(input1_cpu_meta.data_ptr),
                         reinterpret_cast<const float*>(input2_cpu_meta.data_ptr),
                         *reinterpret_cast<const float*>(&scale1),
                         *reinterpret_cast<const float*>(&scale2),
                         reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, half>::value) {
      type_str = "half";
      MulThenAddHalfRef(m, n, reinterpret_cast<const half*>(input1_cpu_meta.data_ptr),
                        reinterpret_cast<const half*>(input2_cpu_meta.data_ptr),
                        *reinterpret_cast<const half*>(&scale1),
                        *reinterpret_cast<const half*>(&scale2),
                        reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      MulThenAddBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input1_cpu_meta.data_ptr),
                        reinterpret_cast<const __nv_bfloat16*>(input2_cpu_meta.data_ptr),
                        *reinterpret_cast<const __nv_bfloat16*>(&scale1),
                        *reinterpret_cast<const __nv_bfloat16*>(&scale2),
                        reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    }

    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    InvokeMulThenAdd<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                        reinterpret_cast<const T*>(input1_meta.data_ptr),
                        reinterpret_cast<const T*>(input2_meta.data_ptr),
                        scale1, scale2, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Use different tolerance for different data types
    float abs_tol = 1e-5f, rel_tol = 1e-5f;
    if (std::is_same<T, half>::value) {
      abs_tol = 1e-3f; rel_tol = 1e-3f;  // Relaxed tolerance for half precision
    }
    EXPECT_TRUE(CheckResult<T>("mul_then_add_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, abs_tol, rel_tol));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(input2_cpu_meta);
    DeleteBuffer(input1_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input1_meta);
    DeleteBuffer(input2_meta);
  }
};

// Test cases for AddThenMul
TEST_F(NvidiaAddMulTestSuit, FloatAddThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_float) {
    TestAddThenMul<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, HalfAddThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_half) {
    TestAddThenMul<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, Bf16AddThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_bf16) {
    TestAddThenMul<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

// Test cases for AddMulSecond
TEST_F(NvidiaAddMulTestSuit, FloatAddMulSecondTest) {
  for (const auto& m_n_pair : m_n_pairs_float) {
    TestAddMulSecond<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, HalfAddMulSecondTest) {
  for (const auto& m_n_pair : m_n_pairs_half) {
    TestAddMulSecond<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, Bf16AddMulSecondTest) {
  for (const auto& m_n_pair : m_n_pairs_bf16) {
    TestAddMulSecond<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

// Test cases for AddBiasThenMul
TEST_F(NvidiaAddMulTestSuit, FloatAddBiasThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_float) {
    TestAddBiasThenMul<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, HalfAddBiasThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_half) {
    TestAddBiasThenMul<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, Bf16AddBiasThenMulTest) {
  for (const auto& m_n_pair : m_n_pairs_bf16) {
    TestAddBiasThenMul<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

// Test cases for MulThenAdd
TEST_F(NvidiaAddMulTestSuit, FloatMulThenAddTest) {
  for (const auto& m_n_pair : m_n_pairs_float) {
    TestMulThenAdd<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, HalfMulThenAddTest) {
  for (const auto& m_n_pair : m_n_pairs_half) {
    TestMulThenAdd<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(NvidiaAddMulTestSuit, Bf16MulThenAddTest) {
  for (const auto& m_n_pair : m_n_pairs_bf16) {
    TestMulThenAdd<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels 