/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/add/add.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaAddTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{2048, 7168}, {60, 7168}, {30, 7168}, {8, 7168}, {7, 7169},
                                                      {4, 256},     {2, 128},   {1, 128},   {128, 1},  {4, 1},
                                                      {1, 4},       {3, 1},     {1, 3},     {1, 1}};

 protected:
  void AddBiasResidualFloatRef(const size_t m, const size_t n, const float* input_a, const float* input_b,
                               const float* bias, float* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        float bias_value = 0.0f;
        if (bias) {
          bias_value = bias[n_idx];
        }
        output[m_idx * n + n_idx] = input_a[m_idx * n + n_idx] + input_b[m_idx * n + n_idx] + bias_value;
      }
    }
  }

  void AddBiasResidualHalfRef(const size_t m, const size_t n, const half* input_a, const half* input_b,
                              const half* bias, half* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        if (bias) {
          output[m_idx * n + n_idx] =
              (half)(((half_float::half)input_a[m_idx * n + n_idx]) + ((half_float::half)input_b[m_idx * n + n_idx]) +
                     ((half_float::half)bias[n_idx]));
        } else {
          output[m_idx * n + n_idx] =
              (half)(((half_float::half)input_a[m_idx * n + n_idx]) + ((half_float::half)input_b[m_idx * n + n_idx]));
        }
      }
    }
  }

  void AddBiasResidualBf16Ref(const size_t m, const size_t n, const __nv_bfloat16* input_a,
                              const __nv_bfloat16* input_b, const __nv_bfloat16* bias, __nv_bfloat16* output) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
      for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        float bias_value = 0.0f;
        if (bias) {
          bias_value = CastBFloat16ToFloat32(static_cast<uint16_t>(bias[n_idx]));
        }
        float a = CastBFloat16ToFloat32(static_cast<uint16_t>(input_a[m_idx * n + n_idx]));
        float b = CastBFloat16ToFloat32(static_cast<uint16_t>(input_b[m_idx * n + n_idx]));
        output[m_idx * n + n_idx] = __float2bfloat16(a + b + bias_value);
      }
    }
  }

  template <typename T>
  void TestAddBiasResidual(const size_t m, const size_t n) {
    // build input and output
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                             /*is_random_init*/ false);
    BufferMeta input_a_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
    BufferMeta input_b_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
    BufferMeta input_bias_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                                 /*is_random_init*/ true);
    BufferMeta output_cpu_meta = CopyToHost<T>(output_meta);
    BufferMeta input_a_cpu_meta = CopyToHost<T>(input_a_meta);
    BufferMeta input_b_cpu_meta = CopyToHost<T>(input_b_meta);
    BufferMeta input_cpu_bias_meta = CopyToHost<T>(input_bias_meta);

    // test add residual without bias
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
      AddBiasResidualHalfRef(m, n, reinterpret_cast<const half*>(input_a_cpu_meta.data_ptr),
                             reinterpret_cast<const half*>(input_b_cpu_meta.data_ptr), nullptr,
                             reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      AddBiasResidualBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input_a_cpu_meta.data_ptr),
                             reinterpret_cast<const __nv_bfloat16*>(input_b_cpu_meta.data_ptr), nullptr,
                             reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, float>::value) {
      type_str = "float";
      AddBiasResidualFloatRef(m, n, reinterpret_cast<const float*>(input_a_cpu_meta.data_ptr),
                              reinterpret_cast<const float*>(input_b_cpu_meta.data_ptr), nullptr,
                              reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    }
    BufferMeta output_ref_meta = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    InvokeAddBiasResidual<T>(
        reinterpret_cast<T*>(output_meta.data_ptr), reinterpret_cast<const T*>(input_a_meta.data_ptr),
        reinterpret_cast<const T*>(input_b_meta.data_ptr), nullptr, nullptr, nullptr, nullptr, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    EXPECT_TRUE(CheckResult<T>("add_bias_residual_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, 1e-5f, 1e-5f));

    // test add residual with bias
    if (std::is_same<T, half>::value) {
      type_str = "half";
      AddBiasResidualHalfRef(m, n, reinterpret_cast<const half*>(input_a_cpu_meta.data_ptr),
                             reinterpret_cast<const half*>(input_b_cpu_meta.data_ptr),
                             reinterpret_cast<const half*>(input_cpu_bias_meta.data_ptr),
                             reinterpret_cast<half*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      AddBiasResidualBf16Ref(m, n, reinterpret_cast<const __nv_bfloat16*>(input_a_cpu_meta.data_ptr),
                             reinterpret_cast<const __nv_bfloat16*>(input_b_cpu_meta.data_ptr),
                             reinterpret_cast<const __nv_bfloat16*>(input_cpu_bias_meta.data_ptr),
                             reinterpret_cast<__nv_bfloat16*>(output_cpu_meta.data_ptr));
    } else if (std::is_same<T, float>::value) {
      type_str = "float";
      AddBiasResidualFloatRef(m, n, reinterpret_cast<const float*>(input_a_cpu_meta.data_ptr),
                              reinterpret_cast<const float*>(input_b_cpu_meta.data_ptr),
                              reinterpret_cast<const float*>(input_cpu_bias_meta.data_ptr),
                              reinterpret_cast<float*>(output_cpu_meta.data_ptr));
    }
    BufferMeta output_ref_meta_with_bias = CopyToDevice<T>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    InvokeAddBiasResidual<T>(reinterpret_cast<T*>(output_meta.data_ptr),
                             reinterpret_cast<const T*>(input_a_meta.data_ptr),
                             reinterpret_cast<const T*>(input_b_meta.data_ptr), nullptr,
                             reinterpret_cast<const T*>(input_bias_meta.data_ptr), nullptr, nullptr, m, n, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    EXPECT_TRUE(CheckResult<T>("add_bias_residual_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta_with_bias, output_meta, 1e-5f, 1e-5f));

    // warmup
    const int kWarmUpRounds = 5;
    for (int i = 0; i < kWarmUpRounds; i++) {
      InvokeAddBiasResidual<T>(
          reinterpret_cast<T*>(output_meta.data_ptr), reinterpret_cast<const T*>(input_a_meta.data_ptr),
          reinterpret_cast<const T*>(input_b_meta.data_ptr), nullptr, nullptr, nullptr, nullptr, m, n, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // performance
    cudaEvent_t start;
    cudaEvent_t stop;
    float time_elapsed_ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    const int kRounds = 10;
    for (int i = 0; i < kRounds; i++) {
      InvokeAddBiasResidual<T>(
          reinterpret_cast<T*>(output_meta.data_ptr), reinterpret_cast<const T*>(input_a_meta.data_ptr),
          reinterpret_cast<const T*>(input_b_meta.data_ptr), nullptr, nullptr, nullptr, nullptr, m, n, stream);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&time_elapsed_ms, start, stop);
    std::cout << "add_bias_residual_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n)
              << " time elapsed : " << time_elapsed_ms / kRounds << " ms ";

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // delete
    DeleteBuffer(output_ref_meta);
    DeleteBuffer(input_b_cpu_meta);
    DeleteBuffer(input_a_cpu_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input_a_meta);
    DeleteBuffer(input_b_meta);
  }
};

TEST_F(LlamaNvidiaAddTestSuit, HalfAddBiasResidualTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddBiasResidual<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaAddTestSuit, FloatAddBiasResidualTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddBiasResidual<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaAddTestSuit, Bf16AddBiasResidualTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddBiasResidual<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels