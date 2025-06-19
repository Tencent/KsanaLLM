/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <thread>

#include "tests/test.h"

#include "ksana_llm/utils/context.h"

namespace ksana_llm {

class ContextTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

#ifdef ENABLE_CUDA
TEST_F(ContextTest, NvidiaInitTest) {
  EXPECT_THROW(
      {
        try {
          std::shared_ptr<Context> context = std::make_shared<Context>(100, 1);
        } catch (const std::runtime_error& e) {
          EXPECT_THAT(e.what(), testing::HasSubstr("tensor_parallel_size should not bigger than devices num:"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(ContextTest, NvidiaCommonTest) {
  constexpr size_t tensor_parallel_size = 2;
  constexpr int attn_data_parallel_size = 1;
  std::shared_ptr<Context> context = std::make_shared<Context>(tensor_parallel_size, attn_data_parallel_size);
  size_t total_rank_num = tensor_parallel_size;

  EXPECT_EQ(context->GetComputeStreams().size(), total_rank_num);
  EXPECT_EQ(context->ext->GetMemoryPools().size(), total_rank_num);
  EXPECT_EQ(context->GetMemoryManageStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetH2DStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetD2HStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetD2DStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetCommStreams().size(), total_rank_num);
  EXPECT_EQ(context->ext->GetCublasHandles().size(), total_rank_num);
  EXPECT_EQ(context->ext->GetCublasLtHandles().size(), total_rank_num);
  EXPECT_EQ(context->ext->GetNCCLParam().size(), total_rank_num);

  EXPECT_EQ(context->GetTensorParallelSize(), tensor_parallel_size);

  for (size_t rank_idx = 0; rank_idx < total_rank_num; ++rank_idx) {
    // check stream valid
    StreamSynchronize(context->GetComputeStreams()[rank_idx]);
    StreamSynchronize(context->GetMemoryManageStreams()[rank_idx]);
    StreamSynchronize(context->GetH2DStreams()[rank_idx]);
    StreamSynchronize(context->GetD2HStreams()[rank_idx]);
    StreamSynchronize(context->GetD2DStreams()[rank_idx]);
    StreamSynchronize(context->GetCommStreams()[rank_idx]);

    // check nccl
    EXPECT_NE(context->ext->GetNCCLParam()[rank_idx].nccl_comm, nullptr);
    EXPECT_GE(context->ext->GetNCCLParam()[rank_idx].rank, 0);
    EXPECT_GE(context->ext->GetNCCLParam()[rank_idx].world_size, 1);

    // cublas
    int cublas_ver = -1;
    CUDA_CHECK(cublasGetVersion(context->ext->GetCublasHandles()[rank_idx], &cublas_ver));
    EXPECT_NE(cublas_ver, -1);
    EXPECT_NE(context->ext->GetCublasLtHandles()[rank_idx], nullptr);
  }
}
#endif

}  // namespace ksana_llm
