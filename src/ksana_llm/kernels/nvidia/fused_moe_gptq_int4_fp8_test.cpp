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

class FusedMoeGptqInt4Fp8KernelTestSuit : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    torch::manual_seed(42);
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

  void TestDequantKernel(const size_t num_experts, const size_t n, const size_t k) {
    torch::Tensor qweight = torch::randint(0, 256, {num_experts, n, k / 2},
                                           torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kUInt8));
    torch::Tensor weight = torch::empty({num_experts, n, k},
                                        torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn));

    DequantInt4Fp8(stream, reinterpret_cast<void*>(weight.data_ptr()),
                   reinterpret_cast<const void*>(qweight.data_ptr()), num_experts * n * k / 2);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    torch::Tensor B = qweight.clone();
    torch::Tensor DB = B.unsqueeze(-1).repeat({1, 1, 1, 2});
    DB.index_put_({torch::indexing::Ellipsis, 1},
                  torch::bitwise_right_shift(DB.index({torch::indexing::Ellipsis, 1}), 4));
    DB = torch::bitwise_and(DB, 0xF);
    DB = DB.to(torch::kFloat32);
    DB = DB - 8;
    DB = DB.to(torch::kFloat8_e4m3fn);
    DB = DB.reshape({DB.size(0), DB.size(1), -1});
    DB = DB.contiguous();

    EXPECT_TRUE(torch::allclose(DB.to(torch::kFloat32), weight.to(torch::kFloat32)));
  }

  template <typename T>
  void TestFusedMoeGptqInt4Fp8Kernel(int m, int n, int k, double err_ration) {
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
    torch::Tensor d_a =
        torch::randn({m, k}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_qb = torch::randint(0, 256, {num_experts, n, k / pack_factor},
                                        torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kUInt8));
    torch::Tensor d_qb_scale =
        torch::randn({num_experts, n, k / group_size},
                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));

    std::unordered_map<std::string, int> config = {
        {"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 1}};

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

    torch::Tensor d_16o =
        torch::empty({m, topk, n}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));
    torch::Tensor d_8o =
        torch::empty({m, topk, n}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>()));

    auto quant16_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqAwqKernel<T>(
          d_a.data_ptr(), d_qb.data_ptr(), d_16o.data_ptr(), d_qb_scale.data_ptr(), nullptr, nullptr,
          d_sorted_token_ids.data_ptr(), d_expert_ids.data_ptr(), d_num_tokens_post_padded.data_ptr(), n, k, em, numel,
          k, 1, n * k / pack_factor, 1, k / pack_factor, n, 1, n * k / group_size, 1, k / group_size,
          n / pack_factor * k / group_size, 1, k / group_size, mul_routed_weight, topk, has_zp, weight_bits, group_size,
          config, stream);
    };
    float quant16_time = MeasureCudaExecutionTime(quant16_kernel, stream);

    // 激活量化
    torch::Tensor d_qa = torch::empty({m, k}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt8));
    torch::Tensor d_qa_scale =
        torch::empty({m, k / 128}, torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32));
    // 反量化权重
    torch::Tensor d_b = torch::empty({num_experts, n, k},
                                     torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn));

    config["block_size_n"] = 128;
    config["block_size_k"] = 128;
    config["group_size_m"] = 1;
    auto quant8_kernel = [&]() {
      InvokePerTokenGroupQuantFp8E4m3<T>(d_a.data_ptr(), d_qa.data_ptr(), d_qa_scale.data_ptr(), m, k, false, stream);
      DequantInt4Fp8(stream, reinterpret_cast<void*>(d_b.data_ptr()), reinterpret_cast<const void*>(d_qb.data_ptr()),
                     (size_t)num_experts * n * k / 2);
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqInt4Fp8Kernel<T>(
          d_qa.data_ptr(), d_b.data_ptr(), d_8o.data_ptr(), d_qa_scale.data_ptr(), d_qb_scale.data_ptr(), nullptr,
          d_sorted_token_ids.data_ptr(), d_expert_ids.data_ptr(), d_num_tokens_post_padded.data_ptr(), n, k, em, numel,
          mul_routed_weight, topk, group_size, config, stream);
    };
    float quant8_time = MeasureCudaExecutionTime(quant8_kernel, stream);

    printf("FuseMoeGptqInt4Fp8KernelTest 16bit percision cost: %f ms, 8bit percison cost: %f ms\n", quant16_time,
           quant8_time);

    // 结果校验
    d_16o = d_16o.to(torch::kFloat32);
    d_8o = d_8o.to(torch::kFloat32);

    double sum_16o = d_16o.sum().item<double>();
    double sum_8o = d_8o.sum().item<double>();
    printf("sum of 16bit and 8bit: %lf, %lf\n", sum_16o, sum_8o);

    torch::Tensor diff = torch::abs(d_16o - d_8o);
    double err = err_ration * d_16o.max().item<double>();
    diff = diff > err;
    int64_t max_diff = diff.sum().item<int64_t>();
    EXPECT_TRUE(diff.sum().item<int64_t>() == 0);
  }
};

TEST_F(FusedMoeGptqInt4Fp8KernelTestSuit, FuseMoeGptqInt4Fp8KernelTest) {
  if (ValidTestHardware()) {
    TestDequantKernel(128, 3584, 1024);
    TestFusedMoeGptqInt4Fp8Kernel<half>(1024, 7168, 2048, 0.03);
    TestFusedMoeGptqInt4Fp8Kernel<__nv_bfloat16>(1024, 7168, 2048, 0.04);
  } else {
    KLLM_LOG_INFO << "Skipping test bacause SM version is less than 90";
  }
}

}  // namespace ksana_llm
