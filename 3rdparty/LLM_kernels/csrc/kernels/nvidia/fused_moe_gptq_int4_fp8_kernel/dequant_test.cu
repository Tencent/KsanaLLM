/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

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
  void TestDequant(const size_t num_experts, const size_t n, const size_t k) {
    BufferMeta qweight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k / 2}, true);
    BufferMeta weight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_experts, n, k}, false);

    // 这里只测速，精度的校验在kernel那边做
    auto cuda_run = [&]() {
      dequant::dequant_int4_fp8(stream, reinterpret_cast<void*>(weight.data_ptr),
                                reinterpret_cast<const void*>(qweight.data_ptr), num_experts * n * k / 2);
    };
    float time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    size_t FLOPS = num_experts * k * n;
    FLOPS = FLOPS * 3 / 2;
    FLOPS = FLOPS * 1000;
    FLOPS = FLOPS / 1024 / 1024 / 1024;

    printf("Dequant cost: %f ms, memory bandwidth: %lf G/s\n", time, (double)FLOPS / time);
  }
};

TEST_F(NvidiaDequantTestSuit, DequantTest) {
#if defined(ENABLE_COMMON_INT4_FP8_DEQUANT)
  TestDequant(128, 2048, 7168);
#else
  std::cerr << "SM version is lower than 90. skipping dequant kernel." << std::endl;
#endif
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels