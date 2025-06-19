/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "test.h"

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
float CastToFloat(T val);

template <>
float CastToFloat<half>(half val) {
  return __half2float(val);
}

template <>
float CastToFloat<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
T CastFloatToDType(float val);

template <>
half CastFloatToDType<half>(float val) {
  return __half2float(val);
}

template <>
__nv_bfloat16 CastFloatToDType<__nv_bfloat16>(float val) {
  return __bfloat162float(val);
}

class FusedMoeKernelTestSuit : public testing::Test {
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
  template <typename T>
  void FusedMoeCpuReference(const T *a, const T *b, T *c, const float *topk_weights, const int32_t *sorted_token_ids,
                            const int32_t *expert_ids, int m, int n, int k, int num_experts, int topk,
                            bool mul_routed_weight, int block_size_m) {
    std::memset(c, 0, m * topk * n * sizeof(T));

    for (int i = 0; i < m * topk; ++i) {
      if (i >= m * topk) continue;
      int token_id = sorted_token_ids[i];
      if (token_id >= m * topk) continue;
      int token_idx = token_id / topk;
      int block_idx = i / block_size_m;
      if (block_idx >= (m * topk + block_size_m - 1) / block_size_m) continue;

      int expert_id = expert_ids[block_idx];
      if (expert_id >= num_experts) continue;

      for (int n_idx = 0; n_idx < n; ++n_idx) {
        double acc = 0.0;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
          double a_val = static_cast<double>(a[token_idx * k + k_idx]);
          double b_val = static_cast<double>(b[(expert_id * n + n_idx) * k + k_idx]);
          acc += a_val * b_val;
        }

        if (mul_routed_weight) {
          acc *= static_cast<double>(topk_weights[i]);
        }

        c[i * n + n_idx] = static_cast<T>(acc);
      }
    }
  }

  bool IsClose(float a, float b, float rtol = 1e-3, float atol = 1e-5) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
  }

  template <typename T>
  bool TensorsClose(const std::vector<T> &a, const std::vector<T> &b, float rtol = 1e-3, float atol = 1e-5) {
    if (a.size() != b.size()) {
      return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
      if (!IsClose(static_cast<float>(a[i]), static_cast<float>(b[i]), rtol, atol)) {
        return false;
      }
    }

    return true;
  }

  void ValidTestHardware() {
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major < 9) {
      GTEST_SKIP() << "Skipping test because SM version is less than 90";
      return;
    }
  }

  template <typename T>
  void TestFusedMoeKernelCorrectness() {
    int m = 8;
    int k = 128;
    int n = 64;
    int num_experts = 4;
    int topk = 1;
    int numel = m * topk;

    std::unordered_map<std::string, int> config = {
        {"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};

    int em = numel + num_experts * (config["block_size_m"] - 1);
    int max_num_m_blocks = (em + config["block_size_m"] - 1) / config["block_size_m"];

    bool mul_routed_weight = true;
    bool use_fp8_w8a8 = false;
    bool use_int8_w8a16 = false;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int_dist(0, num_experts - 1);

    std::vector<T> h_a(m * k);
    std::vector<T> h_b(num_experts * n * k);
    std::vector<T> h_c(m * topk * n);
    std::vector<T> h_c_ref(m * topk * n);
    std::vector<float> h_topk_weights(m * topk);
    std::vector<int32_t> h_sorted_token_ids(em);
    std::vector<int32_t> h_expert_ids(max_num_m_blocks);
    std::vector<int32_t> h_num_tokens_post_padded(1, numel);

    for (int i = 0; i < m * k; ++i) {
      h_a[i] = CastFloatToDType<T>(dist(gen));
    }
    for (int i = 0; i < num_experts * n * k; ++i) {
      h_b[i] = CastFloatToDType<T>(dist(gen));
    }
    for (int i = 0; i < m * topk; ++i) {
      h_topk_weights[i] = dist(gen) * 0.5f + 0.5f;
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < topk; ++j) {
        int idx = i * topk + j;
        h_sorted_token_ids[idx] = idx;
      }
    }
    for (int i = numel; i < em; ++i) {
      h_sorted_token_ids[i] = numel - 1;
    }
    for (int i = 0; i < max_num_m_blocks; ++i) {
      h_expert_ids[i] = int_dist(gen);
    }
    void *d_a, *d_b, *d_c;
    void *d_topk_weights, *d_sorted_token_ids, *d_expert_ids, *d_num_tokens_post_padded;
    CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, num_experts * n * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, m * topk * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_topk_weights, m * topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sorted_token_ids, em * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_ids, max_num_m_blocks * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_tokens_post_padded, sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), num_experts * n * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topk_weights, h_topk_weights.data(), m * topk * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sorted_token_ids, h_sorted_token_ids.data(), em * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_expert_ids, h_expert_ids.data(), max_num_m_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_num_tokens_post_padded, h_num_tokens_post_padded.data(), sizeof(int32_t), cudaMemcpyHostToDevice));

    Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
        d_a, d_b, d_c, nullptr, nullptr, d_topk_weights, d_sorted_token_ids, d_expert_ids, d_num_tokens_post_padded, n,
        k, em, numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128, 0, 0, mul_routed_weight, topk,
        use_fp8_w8a8, use_int8_w8a16, config, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, m * topk * n * sizeof(T), cudaMemcpyDeviceToHost));
    FusedMoeCpuReference<T>(h_a.data(), h_b.data(), h_c_ref.data(), h_topk_weights.data(), h_sorted_token_ids.data(),
                            h_expert_ids.data(), m, n, k, num_experts, topk, mul_routed_weight, config["block_size_m"]);
    float max_rel_error = 0.0f;
    for (int i = 0; i < m * topk * n; ++i) {
      float gpu_val = CastToFloat<T>(h_c[i]);
      float cpu_val = CastToFloat<T>(h_c_ref[i]);
      if (std::abs(cpu_val) > 1e-5) {
        float rel_error = std::abs((gpu_val - cpu_val) / cpu_val);
        max_rel_error = std::max(max_rel_error, rel_error);
      }
    }
    EXPECT_TRUE(TensorsClose(h_c, h_c_ref, 1.0f, 1.0f));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_topk_weights));
    CUDA_CHECK(cudaFree(d_sorted_token_ids));
    CUDA_CHECK(cudaFree(d_expert_ids));
    CUDA_CHECK(cudaFree(d_num_tokens_post_padded));
  }

  template <typename T>
  void TestFusedMoeKernelPerformance() {
    size_t warmup_rounds = 10;
    size_t run_rounds = 50;
    int m = 256;
    int k = 7168;
    int n = 4096;
    int num_experts = 256;
    int topk = 8;
    int numel = m * topk;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int_dist(0, num_experts - 1);

    std::unordered_map<std::string, int> config = {
        {"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 8}};
    if (m <= num_experts) {
      config = {{"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};
    }

    int em = numel + num_experts * (config["block_size_m"] - 1);
    int max_num_m_blocks = (em + config["block_size_m"] - 1) / config["block_size_m"];

    bool mul_routed_weight = false;
    bool use_fp8_w8a8 = false;
    bool use_int8_w8a16 = false;

    std::vector<int32_t> h_sorted_token_ids(em);
    std::vector<int32_t> h_expert_ids(max_num_m_blocks);
    for (int i = numel; i < em; ++i) {
      h_sorted_token_ids[i] = numel - 1;
    }
    for (int i = 0; i < max_num_m_blocks; ++i) {
      h_expert_ids[i] = int_dist(gen);
    }

    void *d_a;  // [m, k]
    void *d_b;  // [num_experts, n, k]
    void *d_c;  // [m, topk, n]
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)m * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)num_experts * n * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t)m * topk * n * sizeof(T)));

    void *topk_weights;            // [m, topk]
    void *sorted_token_ids;        // [em]
    void *expert_ids;              // [max_num_m_blocks]
    void *num_tokens_post_padded;  // [1]
    CUDA_CHECK(cudaMalloc(&topk_weights, m * topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sorted_token_ids, em * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&expert_ids, max_num_m_blocks * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&num_tokens_post_padded, 1 * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(sorted_token_ids, h_sorted_token_ids.data(), em * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(expert_ids, h_expert_ids.data(), max_num_m_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));

    if (m < config["block_size_m"]) {
      em = std::min(em, m * topk * config["block_size_m"]);
    }

    for (size_t run_it = 0; run_it < warmup_rounds; ++run_it) {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
          d_a, d_b, d_c, nullptr, nullptr, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k, em,
          numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128, 0, 0, mul_routed_weight, topk,
          use_fp8_w8a8, use_int8_w8a16, config, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start;
    cudaEvent_t stop;
    float time_elapsed_ms = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (size_t run_it = 0; run_it < run_rounds; ++run_it) {
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
          d_a, d_b, d_c, nullptr, nullptr, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k, em,
          numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128, 0, 0, mul_routed_weight, topk,
          use_fp8_w8a8, use_int8_w8a16, config, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventElapsedTime(&time_elapsed_ms, start, stop));
    std::cout << "TestFusedMoeKernelPerformance " << (std::is_same<T, half>::value ? "half" : "bfloat16")
              << " time elapsed: " << time_elapsed_ms / run_rounds << " ms" << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(topk_weights));
    CUDA_CHECK(cudaFree(sorted_token_ids));
    CUDA_CHECK(cudaFree(expert_ids));
    CUDA_CHECK(cudaFree(num_tokens_post_padded));
  }
};

TEST_F(FusedMoeKernelTestSuit, FusedMoeKernelFP16Test) {
  ValidTestHardware();
  TestFusedMoeKernelPerformance<half>();
}

TEST_F(FusedMoeKernelTestSuit, FusedMoeKernelBF16Test) {
  ValidTestHardware();
  TestFusedMoeKernelPerformance<__nv_bfloat16>();
}

TEST_F(FusedMoeKernelTestSuit, FusedMoeKernelComprehensiveFP16Test) {
  ValidTestHardware();
  TestFusedMoeKernelCorrectness<half>();
}

TEST_F(FusedMoeKernelTestSuit, FusedMoeKernelComprehensiveBF16Test) {
  ValidTestHardware();
  TestFusedMoeKernelCorrectness<__nv_bfloat16>();
}

}  // namespace ksana_llm
