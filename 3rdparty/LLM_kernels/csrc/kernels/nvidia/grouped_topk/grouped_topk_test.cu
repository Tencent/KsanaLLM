/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"
#include "tests/references/deepseek_v3_grouped_topk.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaGroupedTopkTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // Helper function: Allocate device memory
  void* AllocateDeviceMemory(size_t size) {
    void* ptr = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ptr, size));
    return ptr;
  }

  // Helper function: Free device memory
  void FreeDeviceMemory(void* ptr) {
    if (ptr) {
      CHECK_NVIDIA_CUDA_ERROR(cudaFree(ptr));
    }
  }

  // Helper function: Copy data from host to device
  template <typename T>
  void CopyToDevice(const std::vector<float>& h_data, void* d_data, int size) {
    std::vector<T> h_typed_data(size);
    for (int i = 0; i < size; ++i) {
      h_typed_data[i] = static_cast<T>(h_data[i]);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_data, h_typed_data.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Helper function: Copy data from device to host
  void CopyFromDevice(void* d_data, std::vector<float>& h_data, int size) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
  }

  void CopyFromDeviceInt(void* d_data, std::vector<int32_t>& h_data, int size) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_data.data(), d_data, size * sizeof(int32_t), cudaMemcpyDeviceToHost));
  }

  template <typename T>
  void TestDeepSeekV3GroupedTopk() {
    std::vector<std::vector<int>> configs;
    configs.push_back({2048, 256, 8, 8, 4});  // same moe_config with DS-V3
    configs.push_back({2048, 128, 4, 4, 2});
    for (const auto& config : configs) {
      TestInvokeDeepSeekV3GroupedTopkFunction<T>(config);
    }
  }

  template <typename T>
  void TestInvokeDeepSeekV3GroupedTopkFunction(const std::vector<int>& config) {
    // Create input data
    const int tokens_num = config[0];
    const int num_experts = config[1];
    const int topk = config[2];
    const int num_expert_group = config[3];
    const int topk_group = config[4];
    const float routed_scaling_factor = 2.5f;

    std::vector<float> h_gating_output(tokens_num * num_experts, 0.0f);
    std::mt19937 generator(42);  // fixed seed
    std::uniform_real_distribution<float> distribution(-0.99, 0.99);
    for (size_t i = 0; i < h_gating_output.size(); ++i) {
      h_gating_output[i] = distribution(generator);
    }

    std::vector<float> h_e_bias(num_experts, 0.0f);  // bias for all experts
    std::uniform_real_distribution<float> bias_distribution(-0.01, -0.001);
    for (size_t i = 0; i < h_e_bias.size(); ++i) {
      h_e_bias[i] = bias_distribution(generator);
    }

    // Allocate device memory
    void* d_gating_output = AllocateDeviceMemory(tokens_num * num_experts * sizeof(T));
    void* d_e_bias = AllocateDeviceMemory(num_experts * sizeof(float));
    void* d_topk_weights = AllocateDeviceMemory(tokens_num * topk * sizeof(float));
    void* d_topk_ids = AllocateDeviceMemory(tokens_num * topk * sizeof(int32_t));

    // Copy data to device
    CopyToDevice<T>(h_gating_output, d_gating_output, tokens_num * num_experts);
    CopyToDevice<float>(h_e_bias, d_e_bias, num_experts);

    // Call the function under test
    InvokeDeepSeekV3GroupedTopk<T>(d_gating_output, d_e_bias, routed_scaling_factor, d_topk_weights, d_topk_ids,
                                   tokens_num, num_experts, topk, num_expert_group, topk_group, stream);

    // Synchronize stream
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy results back to host
    std::vector<float> h_topk_weights(tokens_num * topk);
    std::vector<int32_t> h_topk_ids(tokens_num * topk);

    CopyFromDevice(d_topk_weights, h_topk_weights, tokens_num * topk);
    CopyFromDeviceInt(d_topk_ids, h_topk_ids, tokens_num * topk);

    std::vector<int32_t> ref_topk_ids(tokens_num * topk);
    std::vector<float> ref_topk_weights(tokens_num * topk);
    RunDeepSeekV3GroupedTopkRef<T>(
        reinterpret_cast<void*>(h_gating_output.data()), reinterpret_cast<void*>(h_e_bias.data()),
        routed_scaling_factor, reinterpret_cast<void*>(ref_topk_weights.data()),
        reinterpret_cast<void*>(ref_topk_ids.data()), tokens_num, num_experts, topk, num_expert_group, topk_group);

    for (int b_idx = 0; b_idx < tokens_num; ++b_idx) {
      // current result
      std::vector<float> h_topk_weghts_vec(topk, 0.0f);
      std::memcpy(h_topk_weghts_vec.data(), h_topk_weights.data() + b_idx * topk, topk * sizeof(float));
      std::sort(h_topk_weghts_vec.begin(), h_topk_weghts_vec.end());
      // refer result
      std::vector<float> ref_topk_weghts_vec(topk, 0.0f);
      std::memcpy(ref_topk_weghts_vec.data(), ref_topk_weights.data() + b_idx * topk, topk * sizeof(float));
      std::sort(ref_topk_weghts_vec.begin(), ref_topk_weghts_vec.end());

      float sum_val = 0.0f;
      for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
        EXPECT_NEAR(h_topk_weghts_vec[topk_idx], ref_topk_weghts_vec[topk_idx], 3e-4)
            << "Topk weights value error true value: " << ref_topk_weghts_vec[topk_idx]
            << " vs our value: " << h_topk_weghts_vec[topk_idx];
        sum_val += h_topk_weghts_vec[topk_idx];
      }

      EXPECT_NEAR(sum_val, routed_scaling_factor, 3e-4);
    }

    for (int b_idx = 0; b_idx < tokens_num; ++b_idx) {
      // current result
      std::vector<int32_t> h_topk_ids_vec(topk, -1);
      std::memcpy(h_topk_ids_vec.data(), h_topk_ids.data() + b_idx * topk, topk * sizeof(int32_t));
      std::sort(h_topk_ids_vec.begin(), h_topk_ids_vec.end());
      // refer result
      std::vector<int32_t> ref_topk_ids_vec(topk, -1);
      std::memcpy(ref_topk_ids_vec.data(), ref_topk_ids.data() + b_idx * topk, topk * sizeof(int32_t));
      std::sort(ref_topk_ids_vec.begin(), ref_topk_ids_vec.end());

      for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
        // in rare cases, scores of different experts are same
        if (h_topk_ids_vec[topk_idx] != ref_topk_ids_vec[topk_idx]) {
          EXPECT_NEAR(ref_topk_weights[b_idx * topk + topk_idx], h_topk_weights[b_idx * topk + topk_idx], 3e-4);
        } else {
          EXPECT_EQ(h_topk_ids_vec[topk_idx], ref_topk_ids_vec[topk_idx])
              << "Topk ids value error true id value: " << ref_topk_ids_vec[topk_idx]
              << " weight: " << ref_topk_weights[b_idx * topk + topk_idx]
              << " vs our value: " << h_topk_ids_vec[topk_idx]
              << " weight: " << h_topk_weights[b_idx * topk + topk_idx];
        }
      }
    }

    size_t warmup_times = 10;
    size_t test_times = 50;
    for (size_t run_it = 0; run_it < warmup_times; ++run_it) {
      InvokeDeepSeekV3GroupedTopk<T>(d_gating_output, d_e_bias, routed_scaling_factor, d_topk_weights, d_topk_ids,
                                     tokens_num, num_experts, topk, num_expert_group, topk_group, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    cudaEvent_t start;
    cudaEvent_t stop;
    float time_elapsed_ms = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));
    for (size_t run_it = 0; run_it < test_times; ++run_it) {
      InvokeDeepSeekV3GroupedTopk<T>(d_gating_output, d_e_bias, routed_scaling_factor, d_topk_weights, d_topk_ids,
                                     tokens_num, num_experts, topk, num_expert_group, topk_group, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&time_elapsed_ms, start, stop));
    std::cout << "InvokeDeepSeekV3GroupedTopk time elapsed: " << time_elapsed_ms / test_times << " ms" << std::endl;

    // Free device memory
    FreeDeviceMemory(d_gating_output);
    FreeDeviceMemory(d_e_bias);
    FreeDeviceMemory(d_topk_weights);
    FreeDeviceMemory(d_topk_ids);
  }
};

TEST_F(LlamaNvidiaGroupedTopkTestSuit, CommonFloatTest) { TestDeepSeekV3GroupedTopk<float>(); }
TEST_F(LlamaNvidiaGroupedTopkTestSuit, CommonHalfTest) { TestDeepSeekV3GroupedTopk<half>(); }
TEST_F(LlamaNvidiaGroupedTopkTestSuit, CommonBFloat16Test) { TestDeepSeekV3GroupedTopk<__nv_bfloat16>(); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels