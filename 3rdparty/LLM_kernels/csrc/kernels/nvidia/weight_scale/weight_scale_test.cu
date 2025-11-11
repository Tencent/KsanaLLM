/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include "csrc/kernels/nvidia/weight_scale/weight_scale_kernel.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
namespace {

template <typename T>
inline T ConvertFromFloat(float value) {
  return static_cast<T>(value);
}

template <>
inline half ConvertFromFloat<half>(float value) {
  return __float2half(value);
}

#ifdef ENABLE_BF16
template <>
inline __nv_bfloat16 ConvertFromFloat<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}
#endif

template <typename T>
inline float GetTolerance() {
  return 1e-6f;
}

template <>
inline float GetTolerance<half>() {
  return 1e-3f;
}

#ifdef ENABLE_BF16
template <>
inline float GetTolerance<__nv_bfloat16>() {
  return 1e-3f;
}
#endif

}  // namespace

class NvidiaWeightScaleTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T>
  void RunAccuracyTest() {
    constexpr int total_tokens = 3;
    constexpr int n_heads = 4;
    constexpr size_t num_elements = total_tokens * n_heads;

    std::vector<float> host_input(num_elements);
    std::vector<float> host_scales(num_elements);

    for (size_t idx = 0; idx < num_elements; ++idx) {
      host_input[idx] = 0.1f * static_cast<float>(idx + 1);
      host_scales[idx] = 1.0f + 0.05f * static_cast<float>((idx % n_heads) + 1);
    }

    BufferMeta input_host = CreateBuffer<T>(MemoryType::MEMORY_CPU, {static_cast<size_t>(total_tokens), static_cast<size_t>(n_heads)});
    BufferMeta q_scale_host = CreateBuffer<float>(MemoryType::MEMORY_CPU, {static_cast<size_t>(total_tokens), static_cast<size_t>(n_heads)});

    T* input_ptr = reinterpret_cast<T*>(input_host.data_ptr);
    float* scale_ptr = reinterpret_cast<float*>(q_scale_host.data_ptr);
    
    for (size_t idx = 0; idx < num_elements; ++idx) {
      input_ptr[idx] = ConvertFromFloat<T>(host_input[idx]);
      scale_ptr[idx] = host_scales[idx];
    }

    BufferMeta input_device = CopyToDevice<T>(input_host);
    BufferMeta q_scale_device = CopyToDevice<float>(q_scale_host);

    const float n_heads_inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n_heads));
    const float softmax_scale = 0.75f;

    InvokeWeightScale<T>(reinterpret_cast<const T*>(input_device.data_ptr),
                         reinterpret_cast<float*>(q_scale_device.data_ptr),
                         n_heads_inv_sqrt,
                         softmax_scale,
                         total_tokens,
                         n_heads,
                         stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<float> device_output(num_elements);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(device_output.data(), q_scale_device.data_ptr,
                                       sizeof(float) * num_elements, cudaMemcpyDeviceToHost));

    const float atol = GetTolerance<T>();
    for (size_t idx = 0; idx < num_elements; ++idx) {
      const float expected = host_input[idx] * host_scales[idx] * n_heads_inv_sqrt * softmax_scale;
      EXPECT_NEAR(device_output[idx], expected, atol)
          << "Mismatch at index " << idx << " expected " << expected << " got " << device_output[idx];
    }

    DeleteBuffer(input_device);
    DeleteBuffer(q_scale_device);
    DeleteBuffer(input_host);
    DeleteBuffer(q_scale_host);
  }
};

TEST_F(NvidiaWeightScaleTestSuit, WeightScaleFloatAccuracy) { RunAccuracyTest<float>(); }

TEST_F(NvidiaWeightScaleTestSuit, WeightScaleHalfAccuracy) { RunAccuracyTest<half>(); }

#ifdef ENABLE_BF16
TEST_F(NvidiaWeightScaleTestSuit, WeightScaleBFloat16Accuracy) { RunAccuracyTest<__nv_bfloat16>(); }
#endif

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
