/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <iostream>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/deep_gemm/fp8_gemm.cuh"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaCutlassDeepGemmTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    // 固定随机数种子
    torch::manual_seed(0);
    torch::cuda::manual_seed_all(0);
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  std::tuple<torch::Tensor, torch::Tensor> per_token_cast_to_fp8(const torch::Tensor& x) {
    auto m = x.size(0);
    auto n = x.size(1);

    auto x_view = x.view({m, -1, 128});

    auto x_amax = x_view.abs().to(torch::kFloat32).amax(2).view({m, -1}).clamp(1e-4);

    auto scaled_x = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch::kFloat8_e4m3fn).view({m, n});
    auto scales = (x_amax / 448.0).view({m, -1});

    return {scaled_x, scales};
  }

  std::tuple<torch::Tensor, torch::Tensor> per_block_cast_to_fp8(const torch::Tensor& x) {
    auto m = x.size(0);
    auto n = x.size(1);

    auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    auto padded_m = ceil_div(m, 128) * 128;
    auto padded_n = ceil_div(n, 128) * 128;

    auto x_padded = torch::zeros({padded_m, padded_n}, x.options()).contiguous();
    x_padded.slice(0, 0, m).slice(1, 0, n).copy_(x);

    auto x_view = x_padded.view({-1, 128, padded_n / 128, 128});

    auto x_amax = x_view.abs().to(torch::kFloat32).amax({1, 3}, true).clamp(1e-4);

    auto x_scaled = (x_view * (448.0 / x_amax)).to(torch::kFloat8_e4m3fn);

    auto result = x_scaled.view_as(x_padded).slice(0, 0, m).slice(1, 0, n).contiguous();
    auto scales = (x_amax / 448.0).view({x_view.size(0), x_view.size(2)});

    return {result, scales};
  }

  torch::Tensor get_col_major_tma_aligned_tensor(const torch::Tensor& x) { return x.transpose(0, 1).contiguous(); }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> construct(
      int m, int k, int n) {
    auto x = torch::randn({m, k}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16)) / k;
    auto y = torch::randn({n, k}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16)) / k;

    auto out = torch::empty({m, n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

    auto ref_out = torch::matmul(x, y.transpose(0, 1));

    auto [x_fp8, x_scales] = per_token_cast_to_fp8(x);
    auto [y_fp8, y_scales] = per_block_cast_to_fp8(y);

    auto x_scales_aligned = get_col_major_tma_aligned_tensor(x_scales);

    return {x_fp8, x_scales_aligned, y_fp8, y_scales, out, ref_out};
  }

  float calc_diff(torch::Tensor x, torch::Tensor y) {
    auto x_d = x.to(torch::kDouble);
    auto y_d = y.to(torch::kDouble);

    auto denominator = (x_d * x_d + y_d * y_d).sum();

    auto sim = 2.0 * (x_d * y_d).sum() / denominator;

    auto diff_tensor = 1.0 - sim;

    return diff_tensor.item<float>();
  }

  void TestCutlassDeepGemmPrecision(int thread_id) {
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));

    bool dg_valid = deep_gemm::jit::getThreadCompiler(thread_id).isValid();
    EXPECT_TRUE(dg_valid);

    int m = 16;
    int k = 7168;
    int n = 2112;

    torch::Tensor x_fp8, x_scales, y_fp8, y_scales, out, ref_out;
    std::tie(x_fp8, x_scales, y_fp8, y_scales, out, ref_out) = construct(m, k, n);

    constexpr uint32_t block_k = 128;
    constexpr uint32_t num_problems = 1;
    int num_device_sms = 78;

    auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
        deep_gemm::jit::get_best_gemm_config(m, n, k, num_problems, num_device_sms);
    auto runtime = deep_gemm::jit::getThreadCompiler(thread_id).build(
        n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages, best_num_tma_multicast,
        deep_gemm::GemmType::Normal, false, thread_id);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemm(kernel, x_fp8.data_ptr(), k, y_fp8.data_ptr(), k, out.data_ptr(), n,
                       static_cast<float*>(x_scales.data_ptr()), static_cast<float*>(y_scales.data_ptr()), m, n, k,
                       best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
                       deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms,
                       static_cast<uint32_t>(best_smem_size));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    float diff = calc_diff(out, ref_out);
    EXPECT_TRUE(diff < 1e-3);
    EXPECT_TRUE(torch::allclose(out, ref_out, 1e-3, 1e-3));

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  void TestCutlassDeepGemmPerformance(int thread_id, int m, int n, int k, int wamrup, int iters) {
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));

    bool dg_valid = deep_gemm::jit::getThreadCompiler(thread_id).isValid();
    EXPECT_TRUE(dg_valid);

    torch::Tensor x_fp8, x_scales, y_fp8, y_scales, out, ref_out;
    std::tie(x_fp8, x_scales, y_fp8, y_scales, out, ref_out) = construct(m, k, n);

    auto cuda_kernel = [&]() {
      constexpr uint32_t block_k = 128;
      constexpr uint32_t num_problems = 1;
      constexpr int num_device_sms = 78;
      auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
          deep_gemm::jit::get_best_gemm_config(m, n, k, num_problems, num_device_sms);
      auto runtime = deep_gemm::jit::getThreadCompiler(thread_id).build(
          n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages, best_num_tma_multicast,
          deep_gemm::GemmType::Normal, false, thread_id);
      auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
      deep_gemm::runGemm(kernel, x_fp8.data_ptr(), k, y_fp8.data_ptr(), k, out.data_ptr(), n,
                         static_cast<float*>(x_scales.data_ptr()), static_cast<float*>(y_scales.data_ptr()), m, n, k,
                         best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
                         deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms,
                         static_cast<uint32_t>(best_smem_size));
    };
    float kernel_time = MeasureCudaExecutionTime(cuda_kernel, stream, wamrup, iters);

    std::cout << fmt::format("DeepGemm mnk({},{},{}) cost: {} ms", m, n, k, kernel_time) << std::endl;

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  void TestCutlassDeepGemmSwapABPrecision(int thread_id) {
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));

    bool dg_valid = deep_gemm::jit::getThreadCompiler(thread_id).isValid();
    EXPECT_TRUE(dg_valid);

    int m = 16;
    int k = 7168;
    int n = 2112;

    torch::Tensor x_fp8, x_scales, y_fp8, y_scales, out, ref_out;
    std::tie(x_fp8, x_scales, y_fp8, y_scales, out, ref_out) = construct(m, k, n);

    constexpr uint32_t block_k = 128;
    constexpr uint32_t num_problems = 1;
    int num_device_sms = 78;

    auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
        deep_gemm::jit::get_best_gemm_config(n, m, k, num_problems, num_device_sms, false, true);
    auto runtime = deep_gemm::jit::getThreadCompiler(thread_id).build(
        n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages, best_num_tma_multicast,
        deep_gemm::GemmType::Normal, true, thread_id);
    auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
    deep_gemm::runGemmSwapAB(kernel, y_fp8.data_ptr(), k, x_fp8.data_ptr(), k, out.data_ptr(), n,
                             static_cast<float*>(y_scales.data_ptr()), static_cast<float*>(x_scales.data_ptr()), n, m,
                             k, best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
                             deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms,
                             static_cast<uint32_t>(best_smem_size));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    float diff = calc_diff(out, ref_out);
    EXPECT_TRUE(diff < 1e-3);
    EXPECT_TRUE(torch::allclose(out, ref_out, 1e-3, 1e-3));

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  }

  void TestCutlassDeepGemmSwapABPerformance(int thread_id, int m, int n, int k, int wamrup, int iters) {
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));

    bool dg_valid = deep_gemm::jit::getThreadCompiler(thread_id).isValid();
    EXPECT_TRUE(dg_valid);

    torch::Tensor x_fp8, x_scales, y_fp8, y_scales, out, ref_out;
    std::tie(x_fp8, x_scales, y_fp8, y_scales, out, ref_out) = construct(m, k, n);

    auto cuda_kernel = [&]() {
      constexpr uint32_t block_k = 128;
      constexpr uint32_t num_problems = 1;
      constexpr int num_device_sms = 78;
      auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
          deep_gemm::jit::get_best_gemm_config(n, m, k, num_problems, num_device_sms, false, true);
      auto runtime = deep_gemm::jit::getThreadCompiler(thread_id).build(
          n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages, best_num_tma_multicast,
          deep_gemm::GemmType::Normal, true, thread_id);
      auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());
      deep_gemm::runGemmSwapAB(kernel, y_fp8.data_ptr(), k, x_fp8.data_ptr(), k, out.data_ptr(), n,
                               static_cast<float*>(y_scales.data_ptr()), static_cast<float*>(x_scales.data_ptr()), n, m,
                               k, best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
                               deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms,
                               static_cast<uint32_t>(best_smem_size));
    };
    float kernel_time = MeasureCudaExecutionTime(cuda_kernel, stream, 10, 100);

    std::cout << fmt::format("DeepGemm swapAB mnk({},{},{}) cost: {} ms", m, n, k, kernel_time) << std::endl;

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  }
};

TEST_F(NvidiaCutlassDeepGemmTestSuit, TestCutlassDeepGemm) {
  TestCutlassDeepGemmPrecision(0);

  int k = 7168;
  int n = 2112;
  for (uint32_t m = 4; m <= 8192; m *= 2) {
    TestCutlassDeepGemmPerformance(0, m, n, k, 10, 100);
  }
}

TEST_F(NvidiaCutlassDeepGemmTestSuit, TestCutlassDeepGemmSwapAB) {
  TestCutlassDeepGemmSwapABPrecision(0);

  int k = 7168;
  int n = 2112;
  for (uint32_t m = 4; m <= 8192; m *= 2) {
    TestCutlassDeepGemmSwapABPerformance(0, m, n, k, 10, 100);
  }
}

TEST_F(NvidiaCutlassDeepGemmTestSuit, TestCutlassDeepGemmMulti) {
  int device_count = -1;
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  if (device_count == 1) {
    GTEST_SKIP_("This test need more than 1 GPU");
  }

  std::vector<std::shared_ptr<std::thread>> run_threads;
  for (int cur_rank = 0; cur_rank < device_count; ++cur_rank) {
    run_threads.emplace_back(std::make_shared<std::thread>([this, cur_rank]() {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      TestCutlassDeepGemmPerformance(cur_rank, 1024, 2112, 7168, 100, 5000);
      TestCutlassDeepGemmSwapABPerformance(cur_rank, 1024, 2112, 7168, 100, 5000);
    }));
  }

  for (int cur_rank = 0; cur_rank < device_count; ++cur_rank) {
    run_threads[cur_rank]->join();
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
