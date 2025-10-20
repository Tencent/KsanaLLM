/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <torch/torch.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "dequant.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaDequantTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const size_t warmup = 100;
  const size_t iters = 1000;

 protected:
  void TestUInt4DequantPerformance(const size_t num_experts, const size_t n, const size_t k) {
    BufferMeta qweight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k / 2}, true);
    BufferMeta weight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k}, false);

    auto cuda_run = [&]() {
      dequant::dequant_uint4_fp8_launcher(stream, weight.data_ptr, qweight.data_ptr, num_experts * n * k / 2);
    };
    float time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    size_t FLOPS = num_experts * k * n;
    FLOPS = FLOPS * 3 / 2;
    FLOPS = FLOPS * 1000;
    FLOPS = FLOPS / 1024 / 1024 / 1024;

    printf("UInt4 Dequant cost: %f ms, memory bandwidth: %lf G/s\n", time, (double)FLOPS / time);
  }

  void TestUInt4DequantPrecision(const size_t num_experts, const size_t n, const size_t k) {
    torch::Tensor qweight = torch::randint(
        0, 256, {static_cast<int32_t>(num_experts), static_cast<int32_t>(n), static_cast<int32_t>(k / 2)},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
    torch::Tensor weight =
        torch::empty({static_cast<int32_t>(num_experts), static_cast<int32_t>(n), static_cast<int32_t>(k)},
                     torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat8_e4m3fn));

    llm_kernels::nvidia::dequant::dequant_uint4_fp8_launcher(stream, weight.data_ptr(), qweight.data_ptr(),
                                                             num_experts * n * k / 2);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

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

  void TestInt4DequantPerformance(const size_t num_experts, const size_t n, const size_t k) {
    BufferMeta qweight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k / 2}, true);
    BufferMeta weight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k}, false);

    auto cuda_run = [&]() {
      dequant::dequant_int4_fp8_launcher(stream, weight.data_ptr, qweight.data_ptr, num_experts * n * k / 2);
    };
    float time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    size_t FLOPS = num_experts * k * n;
    FLOPS = FLOPS * 3 / 2;
    FLOPS = FLOPS * 1000;
    FLOPS = FLOPS / 1024 / 1024 / 1024;

    printf("Int4 Dequant cost: %f ms, memory bandwidth: %lf G/s\n", time, (double)FLOPS / time);
  }
};

TEST_F(NvidiaDequantTestSuit, UInt4DequantTest) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  TestUInt4DequantPerformance(32, 2048, 7168);
  TestUInt4DequantPrecision(32, 2048, 7168);
#else
  std::cerr << "SM version is lower than 90. skipping dequant kernel." << std::endl;
#endif
}

TEST_F(NvidiaDequantTestSuit, Int4DequantTest) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  TestInt4DequantPerformance(32, 2048, 7168);
#else
  std::cerr << "SM version is lower than 90. skipping dequant kernel." << std::endl;
#endif
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels