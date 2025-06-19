/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <chrono>
#include <filesystem>
#include <random>
#include <vector>

#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "test.h"

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace fs = std::filesystem;

namespace ksana_llm {

class PerTokenGroupQuantFP8TestSuit : public testing::Test {
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

  template <typename Func>
  double MeasureExecutionTime(Func &&func, int num_iterations = 100) {
    // 预热
    for (int i = 0; i < 10; ++i) {
      func();
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
      func();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / num_iterations;
  }

  template <typename T>
  void TestPerTokenGroupQuantFP8(int m, int n, bool col_major_scale) {
    T *d_data;
    uint8_t *d_q;
    float *d_s;

    size_t data_size = m * n * sizeof(T);
    size_t q_size = m * n * sizeof(uint8_t);
    size_t s_size;

    if (col_major_scale) {
      s_size = (n / 128) * m * sizeof(float);
    } else {
      s_size = m * (n / 128) * sizeof(float);
    }

    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_q, q_size));
    CUDA_CHECK(cudaMalloc(&d_s, s_size));

    // random input
    std::vector<float> h_data_float(m * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < m * n; ++i) {
      h_data_float[i] = dis(gen);
    }

    // convert into target dtype
    if (std::is_same<T, float>::value) {
      CUDA_CHECK(cudaMemcpy(d_data, h_data_float.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, half>::value) {
      std::vector<half> h_data_half(m * n);
      for (int i = 0; i < m * n; ++i) {
        h_data_half[i] = __float2half(h_data_float[i]);
      }
      CUDA_CHECK(cudaMemcpy(d_data, h_data_half.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      std::vector<__nv_bfloat16> h_data_bf16(m * n);
      for (int i = 0; i < m * n; ++i) {
        h_data_bf16[i] = __float2bfloat16(h_data_float[i]);
      }
      CUDA_CHECK(cudaMemcpy(d_data, h_data_bf16.data(), data_size, cudaMemcpyHostToDevice));
    }

    // measure cost time
    auto run_kernel = [&]() {
      Singleton<TritonWrapper>::GetInstance()->InvokePerTokenGroupQuantFP8<T>(d_data, d_q, d_s, m, n, col_major_scale,
                                                                              stream);
    };

    double avg_time = MeasureExecutionTime(run_kernel, 10);

    KLLM_LOG_INFO << fmt::format("InvokePerTokenGroupQuantFP8<{}> with shape [{}, {}], {} scales {:.5f} ms",
                                 std::is_same<T, half>::value ? "half" : "bfloat16", m, n,
                                 col_major_scale ? "column_major" : "row_major", avg_time);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_s));
  }

  template <typename T>
  void TestPerTokenGroupQuantFP8Correctness(int m, int n, bool col_major_scale) {
    T *d_data;
    uint8_t *d_q;
    float *d_s;

    size_t data_size = m * n * sizeof(T);
    size_t q_size = m * n * sizeof(uint8_t);
    size_t s_size;
    size_t num_groups = n / 128;

    if (col_major_scale) {
      s_size = num_groups * m * sizeof(float);
    } else {
      s_size = m * num_groups * sizeof(float);
    }

    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_q, q_size));
    CUDA_CHECK(cudaMalloc(&d_s, s_size));

    // set input data
    std::vector<float> h_data_float(m * n);
    for (int i = 0; i < m * n; ++i) {
      h_data_float[i] = i / 100.0f;
    }

    if (std::is_same<T, float>::value) {
      CUDA_CHECK(cudaMemcpy(d_data, h_data_float.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, half>::value) {
      std::vector<half> h_data_half(m * n);
      for (int i = 0; i < m * n; ++i) {
        h_data_half[i] = __float2half(h_data_float[i]);
      }
      CUDA_CHECK(cudaMemcpy(d_data, h_data_half.data(), data_size, cudaMemcpyHostToDevice));
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      std::vector<__nv_bfloat16> h_data_bf16(m * n);
      for (int i = 0; i < m * n; ++i) {
        h_data_bf16[i] = __float2bfloat16(h_data_float[i]);
      }
      CUDA_CHECK(cudaMemcpy(d_data, h_data_bf16.data(), data_size, cudaMemcpyHostToDevice));
    }

    // run kernel
    Singleton<TritonWrapper>::GetInstance()->InvokePerTokenGroupQuantFP8<T>(d_data, d_q, d_s, m, n, col_major_scale,
                                                                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint8_t> h_q(m * n);
    std::vector<float> h_s(m * num_groups);
    CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, q_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s.data(), d_s, s_size, cudaMemcpyDeviceToHost));

    std::vector<float> target_scales = {0.0028, 0.0057};
    std::vector<int> target_q = {
        0,   70,  78,  83,  86,  89,  91,  92,  94,  96,  97,  98,  99,  99,  100, 101, 102, 103, 104, 104, 105, 105,
        106, 106, 107, 107, 107, 108, 108, 109, 109, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 113, 113,
        114, 114, 114, 114, 115, 115, 115, 115, 115, 116, 116, 116, 116, 117, 117, 117, 117, 117, 118, 118, 118, 118,
        119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122,
        122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124,
        124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 118, 118, 118, 118,
        118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
        120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122,
        122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123,
        123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 124, 124,
        124, 124, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125,
        125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126};

    for (int i = 0; i < (m * n) / 128; ++i) {
      EXPECT_NEAR(h_s[i], target_scales[i], 1e-3);
    }
    for (int i = 0; i < m * n; ++i) {
      EXPECT_NEAR(static_cast<int>(h_q[i]), target_q[i], 1);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_s));
  }
};

TEST_F(PerTokenGroupQuantFP8TestSuit, HalfPerTokenGroupQuantFP8Performance) {
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  // test different shapes
  std::vector<std::pair<int, int>> shapes = {
      {32, 2560}, {64, 2560}, {128, 2560}, {256, 2560}, {64, 1280}, {64, 5120}, {128, 8192},
  };

  KLLM_LOG_INFO << "===== Testing InvokePerTokenGroupQuantFP8 with half =====";
  for (const auto &shape : shapes) {
    TestPerTokenGroupQuantFP8<half>(shape.first, shape.second, false);
    TestPerTokenGroupQuantFP8<half>(shape.first, shape.second, true);
  }
}

TEST_F(PerTokenGroupQuantFP8TestSuit, BFloat16PerTokenGroupQuantFP8Performance) {
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  // test different shapes
  std::vector<std::pair<int, int>> shapes = {
      {32, 2560}, {64, 2560}, {128, 2560}, {256, 2560}, {64, 1280}, {64, 5120}, {128, 8192},
  };

  KLLM_LOG_INFO << "===== Testing InvokePerTokenGroupQuantFP8 with half =====";
  for (const auto &shape : shapes) {
    TestPerTokenGroupQuantFP8<__nv_bfloat16>(shape.first, shape.second, false);
    TestPerTokenGroupQuantFP8<__nv_bfloat16>(shape.first, shape.second, true);
  }
}

TEST_F(PerTokenGroupQuantFP8TestSuit, HalfPerTokenGroupQuantFP8Correctness) {
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  // 使用小的形状进行正确性测试
  int m = 2;
  int n = 128;

  KLLM_LOG_INFO << "===== Testing InvokePerTokenGroupQuantFP8 Correctness with half =====";

  KLLM_LOG_INFO << "--- Row-major scales ---";
  TestPerTokenGroupQuantFP8Correctness<half>(m, n, false);

  KLLM_LOG_INFO << "--- Column-major scales ---";
  TestPerTokenGroupQuantFP8Correctness<half>(m, n, true);
}

TEST_F(PerTokenGroupQuantFP8TestSuit, BFloat16PerTokenGroupQuantFP8Correctness) {
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  // 使用小的形状进行正确性测试
  int m = 2;
  int n = 128;

  KLLM_LOG_INFO << "===== Testing InvokePerTokenGroupQuantFP8 Correctness with half =====";

  KLLM_LOG_INFO << "--- Row-major scales ---";
  TestPerTokenGroupQuantFP8Correctness<__nv_bfloat16>(m, n, false);

  KLLM_LOG_INFO << "--- Column-major scales ---";
  TestPerTokenGroupQuantFP8Correctness<__nv_bfloat16>(m, n, true);
}

}  // namespace ksana_llm