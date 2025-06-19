/* Copyright 2024 Tencent Inc.  All rights reserved.
 * Adapted from: https://gist.github.com/whitelok/a88b758297f293f13be326f5bf7e5dcd

==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

#include "ksana_llm/kernels/grouped_topk.h"
#include "tests/references/deepseek_v3_grouped_topk.h"

using namespace ksana_llm;

class InvokeGroupedTopkTestSuit : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 protected:
  int32_t device{0};
  cudaStream_t stream;

 protected:
  template <typename T>
  void CopyToDevice(const std::vector<float>& h_data, void* d_data, int size) {
    std::vector<T> h_typed_data(size);
    for (int i = 0; i < size; ++i) {
      h_typed_data[i] = static_cast<T>(h_data[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_data, h_typed_data.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Helper function: Allocate device memory
  void* AllocateDeviceMemory(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
  }

  // Helper function: Allocate host memory
  void* AllocateHostMemory(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
  }

  // Helper function: Free device memory
  void FreeDeviceMemory(void* ptr) {
    if (ptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  // Helper function: Free Host memory
  void FreeHostMemory(void* ptr) {
    if (ptr) {
      CUDA_CHECK(cudaFreeHost(ptr));
    }
  }

  template <typename T>
  void TestGroupedTopk(const std::string& scoring_func, const bool renormalize, const bool has_bias) {
    std::vector<std::vector<int>> configs;
    configs.push_back({2048, 256, 8, 8, 4});  // same moe_config with DS-V3
    configs.push_back({2048, 128, 4, 4, 2});
    for (const auto& config : configs) {
      TestInvokeGroupedTopk<T>(scoring_func, renormalize, has_bias, config);
    }
  }

  template <typename T>
  void TestInvokeGroupedTopk(const std::string& scoring_func, const bool renormalize, const bool has_bias,
                             const std::vector<int>& config) {
    const int tokens_num = config[0];
    const int num_experts = config[1];
    const int topk = config[2];
    const int num_expert_group = config[3];
    const int topk_group = config[4];
    const float routed_scaling_factor = 2.5f;
    const int rank = 0;

    void* d_gating_output = AllocateDeviceMemory(tokens_num * num_experts * sizeof(T));
    std::vector<float> h_gating_output(tokens_num * num_experts, 0.0f);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-0.99, 0.99);
    for (size_t i = 0; i < h_gating_output.size(); ++i) {
      h_gating_output[i] = distribution(generator);
    }
    CopyToDevice<T>(h_gating_output, d_gating_output, tokens_num * num_experts);

    // Create output buffers
    void* topk_weights_ptr = AllocateDeviceMemory(tokens_num * topk * sizeof(float));
    void* topk_ids_ptr = AllocateDeviceMemory(tokens_num * topk * sizeof(int32_t));

    std::vector<int32_t> ref_topk_ids(tokens_num * topk);
    std::vector<float> ref_topk_weights(tokens_num * topk);

    // Create expert bias, ensuring e_bias is not a null pointer
    void* e_bias = nullptr;
    std::vector<float> h_e_bias(num_experts, 0.0f);
    torch::Tensor e_bias_tensor;
    if (has_bias) {
      auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
      e_bias_tensor = torch::rand({num_experts}, options) * 0.1f - 0.05f;  // Small random bias values
      e_bias = e_bias_tensor.data_ptr();
      CUDA_CHECK(cudaMemcpy(h_e_bias.data(), e_bias, num_experts * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Call the function under test
    InvokeGroupedTopk<T>(d_gating_output, topk_weights_ptr, topk_ids_ptr, tokens_num, num_experts, topk, renormalize,
                         num_expert_group, topk_group, scoring_func, e_bias, routed_scaling_factor, rank, stream);

    // Synchronize stream to ensure operation completion
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Validate results
    // Copy results back to host memory for validation
    float* h_topk_weights = reinterpret_cast<float*>(AllocateHostMemory(tokens_num * topk * sizeof(float)));
    int32_t* h_topk_ids = reinterpret_cast<int32_t*>(AllocateHostMemory(tokens_num * topk * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(h_topk_weights, topk_weights_ptr, tokens_num * topk * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_ids, topk_ids_ptr, tokens_num * topk * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Verify topk_ids values are within valid range
    for (int i = 0; i < tokens_num * topk; ++i) {
      EXPECT_GE(h_topk_ids[i], 0);
      EXPECT_LT(h_topk_ids[i], num_experts);
    }

    llm_kernels::nvidia::test::RunDeepSeekV3GroupedTopkRef<T>(
        reinterpret_cast<void*>(h_gating_output.data()), reinterpret_cast<void*>(h_e_bias.data()),
        routed_scaling_factor, reinterpret_cast<void*>(ref_topk_weights.data()),
        reinterpret_cast<void*>(ref_topk_ids.data()), tokens_num, num_experts, topk, num_expert_group, topk_group);

    for (int b_idx = 0; b_idx < tokens_num; ++b_idx) {
      // current result
      std::vector<float> h_topk_weghts_vec(topk, 0.0f);
      std::memcpy(h_topk_weghts_vec.data(), h_topk_weights + b_idx * topk, topk * sizeof(float));
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

      EXPECT_NEAR(sum_val, routed_scaling_factor, 3e-4) << fmt::format(
          "topk weights sum should be {}, but get topk weights sum val: {}", routed_scaling_factor, sum_val);
    }

    for (int b_idx = 0; b_idx < tokens_num; ++b_idx) {
      // current result
      std::vector<int32_t> h_topk_ids_vec(topk, -1);
      std::memcpy(h_topk_ids_vec.data(), h_topk_ids + b_idx * topk, topk * sizeof(int32_t));
      std::sort(h_topk_ids_vec.begin(), h_topk_ids_vec.end());
      // refer result
      std::vector<int32_t> ref_topk_ids_vec(topk, -1);
      std::memcpy(ref_topk_ids_vec.data(), ref_topk_ids.data() + b_idx * topk, topk * sizeof(int32_t));
      std::sort(ref_topk_ids_vec.begin(), ref_topk_ids_vec.end());

      for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
        // in rare cases, scores of different experts are same
        if (h_topk_ids_vec[topk_idx] != ref_topk_ids_vec[topk_idx]) {
          EXPECT_NEAR(ref_topk_weights[b_idx * topk + topk_idx], h_topk_weights[b_idx * topk + topk_idx], 3e-4)
              << fmt::format("weights should be near same when id is different, true id {} vs our id {}",
                             ref_topk_ids_vec[topk_idx], h_topk_ids_vec[topk_idx]);
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
      InvokeGroupedTopk<T>(d_gating_output, topk_weights_ptr, topk_ids_ptr, tokens_num, num_experts, topk, renormalize,
                           num_expert_group, topk_group, scoring_func, e_bias, routed_scaling_factor, rank, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start;
    cudaEvent_t stop;
    float time_elapsed_ms = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (size_t run_it = 0; run_it < test_times; ++run_it) {
      InvokeGroupedTopk<T>(d_gating_output, topk_weights_ptr, topk_ids_ptr, tokens_num, num_experts, topk, renormalize,
                           num_expert_group, topk_group, scoring_func, e_bias, routed_scaling_factor, rank, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed_ms, start, stop));
    KLLM_LOG_INFO << "InvokeGroupedTopk time elapsed: " << time_elapsed_ms / test_times << " ms";

    // Clean up resources
    FreeDeviceMemory(topk_weights_ptr);
    FreeDeviceMemory(topk_ids_ptr);
    FreeHostMemory(h_topk_weights);
    FreeHostMemory(h_topk_ids);
  }
};

// Test with expert bias using sigmoid activation function
TEST_F(InvokeGroupedTopkTestSuit, TestFloatWithExpertBiasSigmoid) {
  TestGroupedTopk<float>("sigmoid", /*renormalize*/ true, /*has_bias*/ true);
}

// Test with half precision, expert bias, and sigmoid activation function
TEST_F(InvokeGroupedTopkTestSuit, TestHalfWithExpertBiasSigmoid) {
  TestGroupedTopk<half>("sigmoid", /*renormalize*/ true, /*has_bias*/ true);
}

#ifdef ENABLE_BFLOAT16
// Test with bfloat16 precision, expert bias, and sigmoid activation function
TEST_F(InvokeGroupedTopkTestSuit, TestBF16WithExpertBiasSigmoid) {
  TestGroupedTopk<__nv_bfloat16>("sigmoid", /*renormalize*/ true, /*has_bias*/ true);
}
#endif