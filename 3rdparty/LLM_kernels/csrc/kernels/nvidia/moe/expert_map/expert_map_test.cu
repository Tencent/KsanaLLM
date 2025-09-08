/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <torch/torch.h>

#include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class NvidiaMoeExpertMapTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    torch::manual_seed(42);
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

TEST_F(NvidiaMoeExpertMapTestSuit, TestMoeExpertMapPrecision) {
  size_t ep_size = 8;
  size_t expert_num = 256;
  size_t tokens = 256;
  size_t topk = 8;

  auto int32_option = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  for (size_t ep_rank = 0; ep_rank < ep_size; ep_rank++) {
    size_t expert_per_rank = expert_num / ep_size;
    size_t start_expert = ep_rank * expert_per_rank;
    size_t end_expert = start_expert + expert_per_rank;

    torch::Tensor input =
        torch::randint(0, expert_num - 1, {static_cast<int64_t>(tokens), static_cast<int64_t>(topk)}, int32_option);

    torch::Tensor condition = (start_expert <= input) & (input < end_expert);
    torch::Tensor output = torch::where(condition, input - start_expert, -1);

    llm_kernels::nvidia::moe::ExpertMap expert_map(ep_size, ep_rank, expert_num);
    expert_map.InvokeExpertMapInplace(static_cast<int32_t*>(input.data_ptr()), tokens * topk, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    EXPECT_TRUE(torch::allclose(input, output));
  }
}

TEST_F(NvidiaMoeExpertMapTestSuit, TestMoeExpertMapPerformance) {
  cudaEvent_t begin, end;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&end));

  const size_t iters = 1000;
  size_t ep_size = 8;
  size_t ep_rank = 5;
  size_t expert_num = 256;
  size_t topk = 8;

  llm_kernels::nvidia::moe::ExpertMap expert_map(ep_size, ep_rank, expert_num);
  auto int32_option = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  const std::vector<size_t> tokens = {1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  for (size_t token : tokens) {
    torch::Tensor input = torch::randint(
        0, expert_num - 1, {static_cast<int64_t>(iters), static_cast<int64_t>(token), static_cast<int64_t>(topk)},
        int32_option);
    for (size_t i = 0; i < iters; ++i) {
      expert_map.InvokeExpertMapInplace(static_cast<int32_t*>(input[i].data_ptr()), token * topk, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(begin, stream));
    for (size_t i = 0; i < iters; ++i) {
      expert_map.InvokeExpertMapInplace(static_cast<int32_t*>(input[i].data_ptr()), token * topk, stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(end, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(end));
    float cost_time;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&cost_time, begin, end));

    printf("token:%zu, cost: %fms\n", token, cost_time);
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(end));
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
