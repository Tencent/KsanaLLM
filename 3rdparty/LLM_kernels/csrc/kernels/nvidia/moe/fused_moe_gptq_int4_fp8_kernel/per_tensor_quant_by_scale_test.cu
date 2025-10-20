/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <torch/torch.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/moe/fused_moe_gptq_int4_fp8_kernel/per_tensor_quant_by_scale.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaPerTokenQuantByScaleTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const size_t warmup = 100;
  const size_t iters = 1000;

 protected:
  void TestPerTokenQuantByScale(size_t token, size_t hidden_size) {
    auto bfloat16_option = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto float8_option = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);

    torch::Tensor input =
        torch::randn({static_cast<int64_t>(token), static_cast<int64_t>(hidden_size)}, bfloat16_option);
    torch::Tensor scales = torch::full({1}, 0.5, bfloat16_option);
    torch::Tensor output =
        torch::empty({static_cast<int64_t>(token), static_cast<int64_t>(hidden_size)}, float8_option);

    auto cuda_run = [&]() {
      per_tensor_quant_by_scale_launcher<__nv_bfloat16, __nv_fp8_e4m3>(output.data_ptr(), input.data_ptr(),
                                                                       scales.data_ptr(), token * hidden_size, stream);
    };
    float time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    size_t FLOPS = token * hidden_size;
    FLOPS = FLOPS * 3;
    FLOPS = FLOPS * 1000;
    FLOPS = FLOPS / 1024 / 1024 / 1024;
    printf("per_tensor_quant_by_scale cost: %f ms, memory bandwidth: %lf G/s\n", time, (double)FLOPS / time);

    torch::Tensor ref = (input * scales).to(torch::kFloat8_e4m3fn);
    EXPECT_TRUE(torch::allclose(output.to(torch::kFloat32), ref.to(torch::kFloat32)));
  }
};

TEST_F(NvidiaPerTokenQuantByScaleTestSuit, PerTokenQuantByScaleTest) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  const std::vector<size_t> tokens = {1, 128, 256, 512, 1024};
  for (size_t token : tokens) {
    TestPerTokenQuantByScale(token, 7168);
  }
#else
  std::cerr << "SM version is lower than 90. skipping per_tensor_quant_by_scale kernel." << std::endl;
#endif
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels