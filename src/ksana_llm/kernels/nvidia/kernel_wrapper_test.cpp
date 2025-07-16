/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include "test.h"

#include "3rdparty/half/include/half.hpp"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"

#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/moe/moe.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/kernels/nvidia/triton_wrapper.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/singleton.h"

#include "ksana_llm/utils/tensor.h"

namespace fs = std::filesystem;

namespace ksana_llm {

// 使用CUDA事件来测量GPU操作的耗时
class CudaEventTimer {
 public:
  CudaEventTimer(const std::string& name, cudaStream_t stream, size_t flops = 0) : name_(name), stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, stream_);
    flops_ = flops;
  }

  ~CudaEventTimer() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    std::string flops_str = flops_ != 0 ? fmt::format(", {} TFlops", flops_ / milliseconds / 1e9) : "";
    KLLM_LOG_INFO << fmt::format("{} cost: {} ms{}", name_, milliseconds, flops_str);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

 private:
  std::string name_;
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  size_t flops_;
};

class KernelWrapperTest : public testing::Test {
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
  int32_t device{-1};
  cudaStream_t stream;
};

float fp8_e4m3fn_to_float(char* in, std::vector<float>& out, int n) {
  for (int i = 0; i < n; i++) {
    const int SIGN_MASK = 0x80;  // 1000 0000
    const int EXP_MASK = 0x78;   // 0111 1000
    const int MANT_MASK = 0x07;  // 0000 0111

    const int EXP_BIAS = 7;
    uint8_t data = (uint8_t)in[i];
    // 提取符号位
    uint8_t sign = (data & SIGN_MASK) >> 7;

    // 提取指数位
    uint8_t exponent = (data & EXP_MASK) >> 3;

    // 提取尾数位
    uint8_t mantissa = data & MANT_MASK;

    float result = 0.0f;

    if (exponent == 0) {
      if (mantissa == 0) {
        // 零
        result = 0.0f;
      } else {
        // 非规范化数（Denormalized Number）
        // 计算公式: (-1)^S * 2^(1 - Bias) * (0 + mantissa / 2^3)
        float mant = static_cast<float>(mantissa) / 8.0f;  // 2^3 = 8
        result = std::ldexp(mant, 1 - EXP_BIAS);
        if (sign) {
          result = -result;
        }
      }
    } else if (exponent == 0x0F && mantissa == 7) {
      // NaN
      result = std::numeric_limits<float>::quiet_NaN();
    } else {
      // 规范化数（Normalized Number）
      // 计算公式: (-1)^S * 2^(E - Bias) * (1 + mantissa / 2^3)
      float mant = 1.0f + static_cast<float>(mantissa) / 8.0f;  // 隐式的 1
      result = std::ldexp(mant, exponent - EXP_BIAS);
      if (sign) {
        result = -result;
      }
    }

    out[i] = result;
  }

  return 0.0;
}

#ifdef ENABLE_FP8_TORCH
TEST_F(KernelWrapperTest, ScaleQuantizeFp8E4m3Test) {
  int m = 256, n = 256;
  size_t num_elements = m * n;
  std::vector<size_t> group_shape = {128, 128};

  std::vector<float> data(num_elements);
  for (size_t i = 0; i < num_elements; i++) {
    data[i] = i * 1.234;
  }

  void* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

  // output
  void* d_data1;
  CUDA_CHECK(cudaMalloc(&d_data1, data.size() * sizeof(char)));

  // scale, not need?
  void* d_data2;
  CUDA_CHECK(cudaMalloc(&d_data2, data.size() / group_shape[0] / group_shape[1] * sizeof(float)));

  ScaledQuantizeFp8E4m3<float>(reinterpret_cast<float*>(d_data), d_data1, reinterpret_cast<float*>(d_data2),
                               group_shape, m, n, 0);

  std::vector<uint8_t> out(num_elements);
  CUDA_CHECK(cudaMemcpy(out.data(), d_data1, num_elements * sizeof(char), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> out_float(num_elements);
  fp8_e4m3fn_to_float(reinterpret_cast<char*>(out.data()), out_float, num_elements);

  //  test data.
  int scale_m = group_shape[0];
  int scale_n = group_shape[1];
  std::vector<float> h_scale(data.size() / group_shape[0] / group_shape[1]);
  std::vector<float> h_out(num_elements);

  // compute scale.
  for (int i = 0; i < m / scale_m; i++) {
    for (int j = 0; j < n / scale_n; j++) {
      // compute block-wise-scale
      float max = 0, min = 100000000.0;
      for (int p = i * scale_m; p < m && p < (i + 1) * scale_m; p++) {
        for (int q = j * scale_n; q < n && q < (j + 1) * scale_n; q++) {
          if (fabsf(data[p * n + q]) > max) {
            max = fabsf(data[p * n + q]);
          }
          if (fabsf(data[p * n + q]) < min) {
            min = fabsf(data[p * n + q]);
          }
        }
      }
      h_scale[i * (n / scale_n) + j] = 448.0 / max;

      // quant
      for (int p = i * scale_m; p < m && p < (i + 1) * scale_m; p++) {
        for (int q = j * scale_n; q < n && q < (j + 1) * scale_n; q++) {
          h_out[p * n + q] = data[p * n + q] * h_scale[i * (n / scale_n) + j];
        }
      }
    }  // end j
  }  // end i

  for (size_t i = 0; i < num_elements; i++) {
    if (fabsf(h_out[i]) > 0) {
      EXPECT_LE(fabsf(fabsf(out_float[i] - h_out[i]) / h_out[i]), 0.3);
    } else {
      EXPECT_LE(fabsf(out_float[i] - h_out[i]), 0.3);
    }
  }
}
#endif

#ifdef ENABLE_CUDA
TEST_F(KernelWrapperTest, MulTest) {
  size_t n = 4;
  std::vector<float> host_a(n, 2);
  std::vector<float> host_b(n, 3);
  std::vector<float> host_c(n, 0);
  float* device_a;
  float* device_b;
  float* device_c;
  CUDA_CHECK(cudaMalloc(&device_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_c, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(device_a, host_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(device_b, host_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  InvokeMul(device_a, device_b, device_c, n, device);
  CUDA_CHECK(cudaMemcpyAsync(host_c.data(), device_c, sizeof(float) * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(host_c[i], host_a[i] * host_b[i]);
  }
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
}

TEST_F(KernelWrapperTest, ReciprocalTest) {
  size_t n = 4;
  std::vector<float> host_in(n, 2);
  std::vector<float> host_out(n, 0);
  float* device_in;
  float* device_out;
  CUDA_CHECK(cudaMalloc(&device_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(device_in, host_in.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Reciprocal(device_out, device_in, n, device);
  CUDA_CHECK(cudaMemcpyAsync(host_out.data(), device_out, sizeof(float) * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(host_out[i], 1.0f / host_in[i]);
  }
  CUDA_CHECK(cudaFree(device_in));
  CUDA_CHECK(cudaFree(device_out));
}

TEST_F(KernelWrapperTest, MaxTest) {
  size_t n = 4;
  std::vector<float> host_a(n, 2);
  std::vector<float> host_b(n, 3);
  std::vector<float> host_c(n, 0);
  float* device_a;
  float* device_b;
  float* device_c;
  CUDA_CHECK(cudaMalloc(&device_a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_c, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpyAsync(device_a, host_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(device_b, host_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Max(device_c, device_a, device_b, n, device);
  CUDA_CHECK(cudaMemcpyAsync(host_c.data(), device_c, sizeof(float) * n, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(host_c[i], std::max(host_a[i], host_b[i]));
  }
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
}

TEST_F(KernelWrapperTest, InvokeFusedMoeTest) {
  // 测试不同的num_tokens和num_experts组合
  std::vector<size_t> test_num_tokens = {32 /*, 512, 1024, 2048, 4096, 8192, 32000*/};
  std::vector<size_t> test_num_experts = {256};
  for (size_t num_tokens : test_num_tokens) {
    for (size_t num_experts : test_num_experts) {
      KLLM_LOG_INFO << fmt::format("===== Testing num_tokens= {}, num_experts={} =====", num_tokens, num_experts);

      // 设置参数
      const int hidden_size = 7168;
      const int inter_size = 2048;
      const int topk = 8;
      const int num_expert_group = 8;
      const int topk_group = 4;
      const bool use_fp8_w8a8 = true;
      const bool use_int8_w8a16 = false;
      const float routed_scaling_factor = 1.0f;
      std::vector<int> block_shape = {128, 128};
      const size_t expert_para_size = 16;
      const int moe_tensor_para_size = 1;
      const size_t num_experts_per_rank = num_experts / expert_para_size;
      const size_t numel = num_tokens * topk;
      const std::unordered_map<std::string, int> config = {{"block_size_m", 64},
                                                           {"block_size_n", block_shape[0]},
                                                           {"block_size_k", block_shape[1]},
                                                           {"group_size_m", 32},
                                                           {"num_warps", 4},
                                                           {"num_stages", 3}};
      const int block_size = config.at("block_size_m");
      const int block_size_n = config.at("block_size_n");
      const int block_size_k = config.at("block_size_k");
      size_t max_num_tokens_padded = numel + num_experts_per_rank * (block_size - 1);
      const size_t max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
      auto int32_options = torch::TensorOptions().device(torch::kCUDA, device).dtype(GetTorchDataType<int32_t>());

      // 分配设备内存
      void* hidden_states;
      void* w1;
      void* w2;
      void* gating_output;
      int* expert_map;
      void* e_bias;
      void* w1_scale;
      void* w2_scale;
      void* a1_q;
      void* a2_q;
      void* a1_scale;
      void* a2_scale;
      void* topk_weights_ptr;
      void* topk_ids_ptr;
      void* fake_topk_ids_ptr;
      void* output_hidden_states;
      void* intermediate_cache1;
      void* intermediate_cache2;
      void* intermediate_cache3;

      CUDA_CHECK(cudaMalloc(&hidden_states, num_tokens * hidden_size * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&w1, num_experts_per_rank * inter_size * 2 * hidden_size * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&w2, num_experts_per_rank * hidden_size * inter_size * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&gating_output, num_tokens * num_experts * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&expert_map, num_experts * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&e_bias, num_experts * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&w1_scale, num_experts_per_rank * inter_size * 2 * hidden_size / 128 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&w2_scale, num_experts_per_rank * hidden_size * inter_size / 128 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&a1_q, num_tokens * hidden_size * sizeof(char)));
      CUDA_CHECK(cudaMalloc(&a2_q, num_tokens * topk * inter_size * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&a1_scale, num_tokens * hidden_size / 128 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&a2_scale, num_tokens * topk * inter_size / 128 * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&topk_weights_ptr, num_tokens * topk * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&topk_ids_ptr, num_tokens * topk * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&fake_topk_ids_ptr, num_tokens * topk * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&output_hidden_states, num_tokens * hidden_size * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&intermediate_cache1, num_tokens * topk * inter_size * 2 * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&intermediate_cache2, num_tokens * topk * inter_size * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&intermediate_cache3, num_tokens * topk * hidden_size * sizeof(half)));

      size_t sum = num_tokens * hidden_size * sizeof(half) + num_experts * inter_size * 2 * hidden_size * sizeof(half) +
                   num_experts * hidden_size * inter_size * sizeof(half) + num_tokens * num_experts * sizeof(float) +
                   num_experts * sizeof(int) + num_experts * sizeof(float) +
                   num_experts * inter_size * 2 * hidden_size / 128 * sizeof(float) +
                   num_experts * hidden_size * inter_size / 128 * sizeof(float) +
                   num_tokens * hidden_size * sizeof(char) + num_tokens * topk * inter_size * sizeof(int) +
                   num_tokens * hidden_size / 128 * sizeof(float) +
                   num_tokens * topk * inter_size / 128 * sizeof(float) + num_tokens * topk * sizeof(half) +
                   num_tokens * topk * sizeof(int) + num_tokens * topk * sizeof(int) +
                   num_tokens * hidden_size * sizeof(half) + num_tokens * topk * inter_size * 2 * sizeof(half) +
                   num_tokens * topk * inter_size * sizeof(half) + num_tokens * topk * hidden_size * sizeof(half);
      KLLM_LOG_INFO << fmt::format("Sum = {} GB", sum / 1024 / 1024 / 1024);

      // 生成随机数据
      std::vector<float> h_hidden_states(num_tokens * hidden_size);
      std::vector<float> h_gating_output(num_tokens * num_experts);
      std::vector<int> h_expert_map(num_experts, num_experts / expert_para_size + 1);
      std::vector<int> h_fake_topk_ids(num_tokens * topk);

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
      std::uniform_int_distribution<int> int_dis(0, num_experts - 1);

      for (size_t i = 0; i < num_tokens * hidden_size; ++i) {
        h_hidden_states[i] = dis(gen);
      }

      for (size_t i = 0; i < num_tokens * num_experts; ++i) {
        h_gating_output[i] = dis(gen);
      }

      for (size_t i = 0; i < num_experts_per_rank; ++i) {
        h_expert_map[i] = i;
      }

      int hit_expert = 0;
      std::vector<int> hit_counts(num_experts, 0);
      for (size_t i = 0; i < num_tokens * topk; ++i) {
        h_fake_topk_ids[i] = int_dis(gen);
        if (h_expert_map[h_fake_topk_ids[i]] < num_experts_per_rank) {
          hit_expert += 1;
          hit_counts[h_expert_map[h_fake_topk_ids[i]]] += 1;
        }
      }

      // 将数据复制到设备
      CUDA_CHECK(cudaMemcpy(hidden_states, h_hidden_states.data(), num_tokens * hidden_size * sizeof(half),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(gating_output, h_gating_output.data(), num_tokens * num_experts * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(expert_map, h_expert_map.data(), num_experts * sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(fake_topk_ids_ptr, h_fake_topk_ids.data(), num_tokens * topk * sizeof(int),
                            cudaMemcpyHostToDevice));

      // 计算各阶段的计算量开销和访存开销
      size_t cal_moe_align_block = static_cast<size_t>(2) + num_experts_per_rank + numel * 2 +
                                   num_experts_per_rank * (1 + numel / num_experts_per_rank * 3) +
                                   num_experts_per_rank * 3 + 1 + num_experts_per_rank * (numel / block_size) +
                                   numel * 5;
      size_t cal_per_token_group_quant1 = num_tokens * hidden_size / 128 * 1288;
      size_t cal_per_token_group_quant2 = num_tokens * inter_size / moe_tensor_para_size / 128 * 1288;
      size_t em = 0;
      for (int i = 0; i < num_experts; ++i) {
        em += (hit_counts[i] + block_size - 1) / block_size * block_size;
      }
      size_t k1 = hidden_size;
      size_t n1 = inter_size * 2;
      size_t cal_fused_moe_1 =
          (22 + 3 * k1 + k1 / block_size_k * 6 + block_size * (10 + 2 * k1) + block_size_n * (7 + 2 * k1) +
           block_size * block_size_n * (2 + 3 * k1 / block_size_k + k1) + block_size * block_size_k +
           block_size_n * block_size_k * 2) *
          (em / block_size * n1 / block_size_n);
      size_t k2 = inter_size;
      size_t n2 = hidden_size;
      size_t cal_fused_moe_2 =
          (22 + 3 * k2 + k2 / block_size_k * 6 + block_size * (11 + 2 * k2) + block_size_n * (7 + 2 * k2) +
           block_size * block_size_n * (3 + 3 * k2 / block_size_k + k2) + block_size * block_size_k +
           block_size_n * block_size_k * 2) *
          (em / block_size * n2 / block_size_n);
      size_t cal_silu = numel * inter_size / moe_tensor_para_size * 10;
      size_t cal_moe_sum = num_tokens * hidden_size * (topk * static_cast<float>(hit_expert) / numel * 3 + 1);

      // 手动组织 MOE 推理流程，插入对应的监控统计
      KLLM_LOG_INFO << fmt::format("Random data expert hit rate: {}",
                                   (static_cast<float>(hit_expert) / (num_tokens * topk)));
      {
        CudaEventTimer total_timer("InvokeFusedMoe Total", stream);
        {
          CudaEventTimer group_topk_timer("InvokeGroupedTopk", stream);
          InvokeGroupedTopk<half>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, num_experts, topk, true,
                                  num_expert_group, topk_group, "sigmoid", e_bias, routed_scaling_factor, device,
                                  stream);
          std::swap(topk_ids_ptr, fake_topk_ids_ptr);
        }
        torch::Tensor sorted_ids, expert_ids, num_tokens_post_pad;
        {
          CudaEventTimer create_moe_tensor("Create MoeAlignBlock output tensor", stream);
          sorted_ids = torch::empty({static_cast<int32_t>(max_num_tokens_padded)}, int32_options);
          sorted_ids.fill_(static_cast<int>(numel));
          expert_ids = torch::empty({static_cast<int32_t>(max_num_m_blocks)}, int32_options);
          expert_ids.fill_(-1);
          num_tokens_post_pad = torch::empty({1}, int32_options);
        }
        {
          CudaEventTimer moe_align_block("AlignBlockSize", stream, cal_moe_align_block);
          if (num_experts_per_rank >= 224) {
            torch::Tensor cumsum = torch::zeros({static_cast<int32_t>(num_experts_per_rank) + 1}, int32_options);
            llm_kernels::nvidia::InvokeSglMoeAlignBlockSize<int32_t>(
                reinterpret_cast<int32_t*>(topk_ids_ptr), sorted_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), num_experts_per_rank,
                block_size, numel, cumsum.data_ptr<int32_t>(), stream);
          } else if (expert_para_size == 1) {
            llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, false>(
                reinterpret_cast<int32_t*>(topk_ids_ptr), sorted_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), expert_map, topk,
                num_experts_per_rank, expert_para_size, block_size, numel, device, stream);
          } else {
            llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, true>(
                reinterpret_cast<int32_t*>(topk_ids_ptr), sorted_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), expert_map, topk,
                num_experts_per_rank, expert_para_size, block_size, numel, device, stream);
          }
        }
        {
          int m = num_tokens;
          int k = hidden_size;
          int n = inter_size * 2;
          {
            CudaEventTimer fused_moe_1("First PerTokenGroupQuantFP8", stream, cal_per_token_group_quant1);
            InvokePerTokenGroupQuantFp8E4m3<half>(hidden_states, a1_q, a1_scale, m, k, false, stream);
          }
          if (m < config.at("block_size_m")) {
            max_num_tokens_padded = std::min(max_num_tokens_padded, num_tokens * topk * config.at("block_size_m"));
          }
          {
            CudaEventTimer fused_moe_1("First FusedMoeKernel", stream, cal_fused_moe_1);
            Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<half>(
                a1_q, w1, intermediate_cache1, a1_scale, w1_scale, topk_weights_ptr, sorted_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), n, k, max_num_tokens_padded,
                numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128, block_shape[0],
                block_shape[1], false, topk, use_fp8_w8a8, use_int8_w8a16, config, stream);
          }
        }
        {
          CudaEventTimer silu_and_mul("InvokeSiluAndMul", stream, cal_silu);
          size_t elements_num = static_cast<size_t>(num_tokens) * topk * inter_size * 2;
          if (expert_para_size == 1) {
            llm_kernels::nvidia::InvokeSiluAndMul<half, false>(
                reinterpret_cast<const half*>(intermediate_cache1), reinterpret_cast<half*>(intermediate_cache2),
                reinterpret_cast<const int*>(topk_ids_ptr), reinterpret_cast<const int*>(expert_map),
                num_experts_per_rank, elements_num, inter_size, stream);
          } else {
            llm_kernels::nvidia::InvokeSiluAndMul<half, true>(
                reinterpret_cast<const half*>(intermediate_cache1), reinterpret_cast<half*>(intermediate_cache2),
                reinterpret_cast<const int*>(topk_ids_ptr), reinterpret_cast<const int*>(expert_map),
                num_experts_per_rank, elements_num, inter_size, stream);
          }
        }
        {
          int m = num_tokens * topk;
          int k = inter_size;
          int n = hidden_size;
          {
            CudaEventTimer fused_moe_2("Second PerTokenGroupQuantFP8", stream, cal_per_token_group_quant2);
            InvokePerTokenGroupQuantFp8E4m3<half>(intermediate_cache2, a2_q, a2_scale, m, k, false, stream);
          }
          if (m < config.at("block_size_m")) {
            max_num_tokens_padded = std::min(max_num_tokens_padded, num_tokens * 1 * config.at("block_size_m"));
          }
          {
            CudaEventTimer fused_moe_2("Second FusedMoeKernel", stream, cal_fused_moe_2);
            Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<half>(
                a2_q, w2, intermediate_cache3, a2_scale, w2_scale, topk_weights_ptr, sorted_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(), n, k, max_num_tokens_padded,
                numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128, block_shape[0],
                block_shape[1], true, 1, use_fp8_w8a8, use_int8_w8a16, config, stream);
          }
        }
        {
          CudaEventTimer fused_moe_1("InvokeMoeSum", stream, cal_moe_sum);
          if (expert_para_size == 1) {
            llm_kernels::nvidia::InvokeMoeSum<half, false>(intermediate_cache3, output_hidden_states, topk_ids_ptr,
                                                           expert_map, num_tokens, num_experts_per_rank, topk,
                                                           hidden_size, stream);
          } else {
            llm_kernels::nvidia::InvokeMoeSum<half, true>(intermediate_cache3, output_hidden_states, topk_ids_ptr,
                                                          expert_map, num_tokens, num_experts_per_rank, topk,
                                                          hidden_size, stream);
          }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      // 清理内存
      CUDA_CHECK(cudaFree(hidden_states));
      CUDA_CHECK(cudaFree(w1));
      CUDA_CHECK(cudaFree(w2));
      CUDA_CHECK(cudaFree(gating_output));
      CUDA_CHECK(cudaFree(expert_map));
      CUDA_CHECK(cudaFree(e_bias));
      CUDA_CHECK(cudaFree(w1_scale));
      CUDA_CHECK(cudaFree(w2_scale));
      CUDA_CHECK(cudaFree(a1_q));
      CUDA_CHECK(cudaFree(a2_q));
      CUDA_CHECK(cudaFree(a1_scale));
      CUDA_CHECK(cudaFree(a2_scale));
      CUDA_CHECK(cudaFree(topk_weights_ptr));
      CUDA_CHECK(cudaFree(topk_ids_ptr));
      CUDA_CHECK(cudaFree(fake_topk_ids_ptr));
      CUDA_CHECK(cudaFree(output_hidden_states));
      CUDA_CHECK(cudaFree(intermediate_cache1));
      CUDA_CHECK(cudaFree(intermediate_cache2));
      CUDA_CHECK(cudaFree(intermediate_cache3));
    }
  }
}

#endif

}  // namespace ksana_llm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
