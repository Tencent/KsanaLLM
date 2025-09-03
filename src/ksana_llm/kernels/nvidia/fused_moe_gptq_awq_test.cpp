/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "test.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/kernels/nvidia/triton_wrapper.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class FusedMoeGptqAwqKernelTestSuit : public testing::Test {
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
  int32_t rank{0};
  int32_t device{-1};
  cudaStream_t stream;

  bool ValidTestHardware() {
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major >= 9;
  }

  void MoeAlignBlockCpu(std::vector<int>& topk_ids, std::vector<int>& expert_ids, std::vector<int>& sorted_ids,
                        std::vector<int>& token_post_pad, int token_num, int topk, int expert_num, int block_size) {
    std::vector<int> cumsum(expert_num + 1);
    std::vector<int> token_cnts(expert_num + 1);
    size_t numel = static_cast<size_t>(token_num) * topk;
    for (size_t i = 0; i < numel; ++i) {
      int expert_id = topk_ids[i];
      if (expert_id >= expert_num) {
        continue;
      }
      token_cnts[expert_id] += 1;
    }
    for (int i = 0; i < expert_num; ++i) {
      cumsum[i + 1] = cumsum[i] + (token_cnts[i] + block_size - 1) / block_size;
      token_cnts[i] = 0;
      for (int j = cumsum[i]; j < cumsum[i + 1]; ++j) {
        expert_ids[j] = i;
      }
    }
    token_post_pad[0] = cumsum[expert_num] * block_size;
    for (size_t i = 0; i < numel; ++i) {
      int expert_id = topk_ids[i];
      if (expert_id >= expert_num) {
        continue;
      }
      int idx = cumsum[expert_id] * block_size + token_cnts[expert_id];
      sorted_ids[idx] = i;
      token_cnts[expert_id] += 1;
    }
  }

  template <typename T>
  void TestFusedMoeGptqAwqKernel() {
    int m = 16;
    int k = 2048;
    int n = 7168;
    int num_experts = 256;
    int topk = 8;
    int numel = m * topk;
    bool mul_routed_weight = false;
    bool has_zp = false;
    int weight_bits = 4;
    int group_size = 128;
    int pack_factor = 8 / weight_bits;

    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    // 生成随机数据
    // void* d_a;         // [m, k]
    // void* d_qb;        // [num_experts, n, k / pack_factor]
    // void* d_qb_scale;  // [num_experts, n, k / group_size]
    // void* d_b;         // [num_experts, n, k]
    void* d_qo;  // [m, topk, n]
    void* d_o;   // [m, topk, n]
    CUDA_CHECK(cudaMalloc(&d_qo, (size_t)m * topk * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_o, (size_t)m * topk * n * sizeof(T)));

    torch::Tensor d_a = torch::randn({static_cast<int64_t>(m) * k},
                                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_qb = torch::randint(0, 256, {static_cast<int64_t>(num_experts) * n * k / pack_factor},
                                        torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kUInt8));
    torch::Tensor d_qb_scale =
        torch::randn({static_cast<int64_t>(num_experts) * n * k / group_size},
                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    // 反量化
    torch::Tensor high_bits = torch::bitwise_right_shift(d_qb, 4);
    torch::Tensor low_bits = torch::bitwise_and(d_qb, 0x0F);
    torch::Tensor unpacked_b = torch::cat({low_bits.unsqueeze(1), high_bits.unsqueeze(1)}, 1).flatten();
    torch::Tensor unpacked_qb_scale = d_qb_scale.unsqueeze(-1).repeat({1, group_size}).flatten();
    torch::Tensor d_b = unpacked_qb_scale * (unpacked_b.to(GetTorchDataType<T>()) - 8);

    std::unordered_map<std::string, int> config = {
        {"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 8}};
    if (m <= num_experts) {
      config = {{"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};
    }

    int em = numel + num_experts * (config["block_size_m"] - 1);
    int max_num_m_blocks = (em + config["block_size_m"] - 1) / config["block_size_m"];

    torch::Tensor topk_ids = torch::randint(0, num_experts, {m * topk}, torch::kInt32);
    std::vector<int32_t> topk_ids_vector(topk_ids.data_ptr<int32_t>(), topk_ids.data_ptr<int32_t>() + topk_ids.numel());
    std::vector<int32_t> expert_ids(max_num_m_blocks, -1);
    std::vector<int32_t> sorted_token_ids(em, numel);
    std::vector<int32_t> num_tokens_post_padded(1);
    // 一定要按照逻辑造数据，随机数据影响性能
    MoeAlignBlockCpu(topk_ids_vector, expert_ids, sorted_token_ids, num_tokens_post_padded, m, topk, num_experts,
                     config["block_size_m"]);

    // 生成MOE必要的相关数据
    // void* d_topk_weights;            // [m, topk]
    // void* d_sorted_token_ids;        // [em]
    // void* d_expert_ids;              // [max_num_m_blocks]
    // void* d_num_tokens_post_padded;  // [1]
    torch::Tensor d_sorted_token_ids =
        torch::from_blob(sorted_token_ids.data(), {em}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));
    torch::Tensor d_expert_ids =
        torch::from_blob(expert_ids.data(), {max_num_m_blocks}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));
    torch::Tensor d_num_tokens_post_padded =
        torch::from_blob(num_tokens_post_padded.data(), {1}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));

    if (m < config["block_size_m"]) {
      em = std::min(em, m * topk * config["block_size_m"]);
    }

    auto quant_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqAwqKernel<T>(
          d_a.data_ptr(), d_qb.data_ptr(), d_qo, d_qb_scale.data_ptr(), nullptr, nullptr, d_sorted_token_ids.data_ptr(),
          d_expert_ids.data_ptr(), d_num_tokens_post_padded.data_ptr(), n, k, em, numel, k, 1, n * k / pack_factor, 1,
          k / pack_factor, n, 1, n * k / group_size, 1, k / group_size, n / pack_factor * k / group_size, 1,
          k / group_size, mul_routed_weight, topk, has_zp, weight_bits, group_size, config, stream);
    };
    float quant_time = MeasureCudaExecutionTime(quant_kernel, stream);

    auto default_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
          d_a.data_ptr(), d_b.data_ptr(), d_o, nullptr, nullptr, nullptr, d_sorted_token_ids.data_ptr(),
          d_expert_ids.data_ptr(), d_num_tokens_post_padded.data_ptr(), n, k, em, numel, k, 1, n * k, 1, k, n, 1,
          k / 128, 1, n / 128 * k / 128, 1, k / 128, 0, 0, mul_routed_weight, topk, false, false, config, stream);
    };
    float default_time = MeasureCudaExecutionTime(default_kernel, stream);

    printf("FuseMoeGptqAwqKernelFP16Test default percision cost: %f ms, quant percison cost: %f ms\n", default_time,
           quant_time);

    // 结果校验
    std::vector<T> h_qo((size_t)m * topk * n);
    std::vector<T> h_o((size_t)m * topk * n);
    CUDA_CHECK(cudaMemcpy(h_qo.data(), d_qo, h_qo.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double sum_qo = 0, sum_o = 0;
    for (size_t idx = 0; idx < h_o.size(); idx++) {
      EXPECT_FLOAT_EQ(static_cast<float>(h_qo[idx]), static_cast<float>(h_o[idx]));
      sum_qo += static_cast<double>(h_qo[idx]);
      sum_o += static_cast<double>(h_o[idx]);
    }
    printf("sum: %lf, %lf\n", sum_qo, sum_o);
  }

  template <typename T>
  void TestMoeWna16Kernel() {
    int m = 16;
    int k = 2048;
    int n = 7168;
    int num_experts = 256;
    int topk = 8;
    int numel = m * topk;
    bool mul_routed_weight = false;
    bool has_zp = false;
    int weight_bits = 4;
    int group_size = 128;
    int pack_factor = 8 / weight_bits;

    // 生成随机数据
    // void* d_a;         // [m, k]
    // void* d_qb;        // [num_experts, n, k / pack_factor]
    // void* d_qb_scale;  // [num_experts, n, k / group_size]
    // void* d_b;         // [num_experts, n, k]
    // void* d_o;         // [m, topk, n]
    torch::Tensor d_a = torch::randn({static_cast<int64_t>(m) * k},
                                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_qb = torch::randint(0, 256, {static_cast<int64_t>(num_experts) * n * k / pack_factor},
                                        torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kUInt8));
    torch::Tensor d_qb_scale =
        torch::randn({static_cast<int64_t>(num_experts) * n * k / group_size},
                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_to = torch::zeros({static_cast<int64_t>(m) * topk * n},
                                      torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_co = torch::zeros_like(d_to);

    std::unordered_map<std::string, int> triton_config = {
        {"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};

    int em = numel + num_experts * (triton_config["block_size_m"] - 1);
    int max_num_m_blocks = (em + triton_config["block_size_m"] - 1) / triton_config["block_size_m"];

    torch::Tensor topk_ids = torch::randint(0, num_experts, {m * topk}, torch::kInt32);
    std::vector<int32_t> topk_ids_vector(topk_ids.data_ptr<int32_t>(), topk_ids.data_ptr<int32_t>() + topk_ids.numel());
    std::vector<int32_t> expert_ids(max_num_m_blocks, -1);
    std::vector<int32_t> sorted_token_ids(em, numel);
    std::vector<int32_t> num_tokens_post_padded(1);
    // 一定要按照逻辑造数据，随机数据影响性能
    MoeAlignBlockCpu(topk_ids_vector, expert_ids, sorted_token_ids, num_tokens_post_padded, m, topk, num_experts,
                     triton_config["block_size_m"]);

    // 生成MOE必要的相关数据
    // void* d_topk_weights;            // [m, topk]
    // void* d_sorted_token_ids;        // [em]
    // void* d_expert_ids;              // [max_num_m_blocks]
    // void* d_num_tokens_post_padded;  // [1]
    torch::Tensor d_sorted_token_ids =
        torch::from_blob(sorted_token_ids.data(), {em}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));
    torch::Tensor d_expert_ids =
        torch::from_blob(expert_ids.data(), {max_num_m_blocks}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));
    torch::Tensor d_num_tokens_post_padded =
        torch::from_blob(num_tokens_post_padded.data(), {1}, torch::TensorOptions().dtype(torch::kInt32))
            .to(torch::Device(torch::kCUDA, rank));

    if (m < triton_config["block_size_m"]) {
      em = std::min(em, m * topk * triton_config["block_size_m"]);
    }
    auto triton_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqAwqKernel<T>(
          d_a.data_ptr(), d_qb.data_ptr(), d_to.data_ptr(), d_qb_scale.data_ptr(), nullptr, nullptr,
          d_sorted_token_ids.data_ptr(), d_expert_ids.data_ptr(), d_num_tokens_post_padded.data_ptr(), n, k, em, numel,
          k, 1, n * k / pack_factor, 1, k / pack_factor, n, 1, n * k / group_size, 1, k / group_size,
          n / pack_factor * k / group_size, 1, k / group_size, mul_routed_weight, topk, has_zp, weight_bits, group_size,
          triton_config, stream);
    };
    float triton_time = MeasureCudaExecutionTime(triton_kernel, stream);

    bool use_moe_wna16_cuda = ShouldMoeWna16UseCuda(m * topk, group_size, num_experts, weight_bits);
    std::unordered_map<std::string, int> cuda_config = {{"block_size_m", std::min(16, m)}};
    UpdateMoeWna16BlockConfig(cuda_config, use_moe_wna16_cuda, false, m * topk, k, n, num_experts, group_size, topk,
                              cuda_config["block_size_m"]);

    int64_t EM = sorted_token_ids.size();
    if (m <= cuda_config["block_size_m"]) {
      EM = std::min(EM, static_cast<int64_t>(m * cuda_config["block_size_m"] * topk));
    }
    const int num_token_blocks = (EM + cuda_config["block_size_m"] - 1) / cuda_config["block_size_m"];

    auto cuda_kernel = [&]() {
      InvokeMoeWna16Gemm<T>(stream, d_co.data_ptr(), d_a.data_ptr(), d_qb.data_ptr(), d_qb_scale.data_ptr(), nullptr,
                            nullptr, d_sorted_token_ids.data_ptr(), d_expert_ids.data_ptr(),
                            d_num_tokens_post_padded.data_ptr(), topk, cuda_config["block_size_m"],
                            cuda_config["block_size_n"], cuda_config["block_size_k"], weight_bits, num_experts, m, n, k,
                            group_size, num_token_blocks);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_kernel, stream);

    printf("TestMoeWna16Kernel triton cost: %f ms, cuda cost: %f ms\n", triton_time, cuda_time);

    // 结果校验
    std::vector<T> h_to((size_t)m * topk * n);
    std::vector<T> h_co((size_t)m * topk * n);
    CUDA_CHECK(cudaMemcpy(h_to.data(), d_to.data_ptr(), h_to.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_co.data(), d_co.data_ptr(), h_co.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double sum_to = 0, sum_co = 0;
    for (size_t idx = 0; idx < h_co.size(); idx++) {
      EXPECT_NEAR(static_cast<float>(h_to[idx]), static_cast<float>(h_co[idx]), 20.0);  // BF16误差大，要给大一点
      sum_to += static_cast<double>(h_to[idx]);
      sum_co += static_cast<double>(h_co[idx]);
    }
    printf("sum: %lf, %lf\n", sum_to, sum_co);
  }
};

TEST_F(FusedMoeGptqAwqKernelTestSuit, FuseMoeGptqAwqKernelTest) {
  if (ValidTestHardware()) {
    TestFusedMoeGptqAwqKernel<half>();
    TestFusedMoeGptqAwqKernel<__nv_bfloat16>();
  } else {
    KLLM_LOG_INFO << "Skipping test bacause SM version is less than 90";
  }
}

TEST_F(FusedMoeGptqAwqKernelTestSuit, MoeWna16KernelTest) {
  if (ValidTestHardware()) {
    TestMoeWna16Kernel<half>();
    TestMoeWna16Kernel<__nv_bfloat16>();
  } else {
    KLLM_LOG_INFO << "Skipping test bacause SM version is less than 90";
  }
}

}  // namespace ksana_llm
