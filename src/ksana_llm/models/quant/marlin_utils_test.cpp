/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/quant/marlin_utils.h"
#include "tests/test.h"

using namespace ksana_llm;

class MarlinUtilsTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1, 1);
    marlin_helper_ = std::make_shared<MarlinUtils>(context_, rank_, bits_, groupsize_);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {}

 protected:
  std::shared_ptr<MarlinUtils> marlin_helper_{nullptr};
  std::shared_ptr<Context> context_{nullptr};
  int rank_{0};
  int bits_{4};
  int groupsize_{128};
};

TEST_F(MarlinUtilsTest, TestMarlinPrepack) {
#ifdef ENABLE_CUDA
  size_t num_rows = 1024, num_cols = 1024;

  // tensor is 2D
  torch::Tensor tensor =
      torch::arange(static_cast<int64_t>(num_rows * num_cols), torch::kInt32).to(torch::Device(torch::kCUDA, rank_));
  tensor = tensor.view({static_cast<int64_t>(num_rows), static_cast<int64_t>(num_cols)});

  torch::Tensor awq = marlin_helper_->PackAwqWeight(tensor);
  torch::Tensor gptq = marlin_helper_->PackGptqWeight(tensor, std::nullopt);

  // shape检查
  EXPECT_TRUE(gptq.size(0) == 512);
  EXPECT_TRUE(gptq.size(1) == 2048);
  EXPECT_TRUE(awq.size(0) == 64);
  EXPECT_TRUE(awq.size(1) == 16384);

  // 数据检查
  gptq = gptq.flatten();
  awq = awq.flatten();

  std::vector<int32_t> gptq_ref_0 = {34816, 286361600, 572688384, 859015168, 16448, 16448, 16448, 16448, 0, 0};
  std::vector<int32_t> awq_ref_0 = {285217024, 857879330,  1430541636, 2003203942, 285217024,
                                    857879330, 1430541636, 2003203942, 285217024,  857879330};
  for (size_t i = 0; i < 10; i++) {
    EXPECT_TRUE(gptq[i].item<int32_t>() == gptq_ref_0[i]);
    EXPECT_TRUE(awq[i].item<int32_t>() == awq_ref_0[i]);
  }

  std::vector<int32_t> gptq_ref_1 = {-1029, -1029, 65535, 65535, 65535, 65535, 0, 0, 0, 0};
  std::vector<int32_t> awq_ref_1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < 10; i++) {
    EXPECT_TRUE(gptq[num_rows * num_cols - 10 + i].item<int32_t>() == gptq_ref_1[i]);
    EXPECT_TRUE(awq[num_rows * num_cols - 10 + i].item<int32_t>() == awq_ref_1[i]);
  }

  // tensor is 3D
  size_t num_experts = 4;
  tensor = tensor.repeat({static_cast<int64_t>(num_experts), 1, 1}).contiguous();

  gptq = marlin_helper_->PackGptqWeight(tensor, std::nullopt);

  // shape检查
  EXPECT_TRUE(gptq.size(0) == 4);
  EXPECT_TRUE(gptq.size(1) == 512);
  EXPECT_TRUE(gptq.size(2) == 2048);

  // 数据检查
  gptq = gptq.view({gptq.size(0), gptq.size(1) * gptq.size(2)});

  for (size_t e = 0; e < num_experts; e++) {
    for (size_t i = 0; i < 10; i++) {
      EXPECT_TRUE(gptq[e][i].item<int32_t>() == gptq_ref_0[i]);
    }
  }

  for (size_t e = 0; e < num_experts; e++) {
    for (size_t i = 0; i < 10; i++) {
      EXPECT_TRUE(gptq[e][num_rows * num_cols - 10 + i].item<int32_t>() == gptq_ref_1[i]);
    }
  }
#endif
}

TEST_F(MarlinUtilsTest, TestMarlinSortGIdx) {
#ifdef ENABLE_CUDA
  size_t gidx_len = 512;

  torch::manual_seed(42);
  // tensor is 1D
  int32_t int32_min = std::numeric_limits<int32_t>::min();
  int32_t int32_max = std::numeric_limits<int32_t>::max();
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, rank_));
  torch::Tensor gidx = torch::randint(int32_min, int32_max, {static_cast<int64_t>(gidx_len)}, options);

  // Note: gidx is modified
  torch::Tensor perm = marlin_helper_->MarlinSortGIdx(gidx);

  // 数据检查
  std::vector<int32_t> gidx_ref_0 = {-2143065743, -2125184002, -2115842831, -2112175817, -2104422457,
                                     -2104281239, -2101887190, -2099697292, -2098007451, -2084918534};
  std::vector<int32_t> perm_ref_0 = {309, 346, 20, 292, 485, 1, 424, 412, 256, 136};
  for (size_t i = 0; i < 10; i++) {
    EXPECT_TRUE(gidx[i].item<int32_t>() == gidx_ref_0[i]);
    EXPECT_TRUE(perm[i].item<int32_t>() == perm_ref_0[i]);
  }

  std::vector<int32_t> gidx_ref_1 = {2092600530, 2099881348, 2111888513, 2112545028, 2114573769,
                                     2115374391, 2121156829, 2124809658, 2139573741, 2146308641};
  std::vector<int32_t> perm_ref_1 = {278, 391, 324, 453, 386, 74, 205, 474, 329, 493};
  for (size_t i = 0; i < 10; i++) {
    EXPECT_TRUE(gidx[gidx_len - 10 + i].item<int32_t>() == gidx_ref_1[i]);
    EXPECT_TRUE(perm[gidx_len - 10 + i].item<int32_t>() == perm_ref_1[i]);
  }

  // tensor is 2D
  size_t num_experts = 4;
  torch::manual_seed(42);
  gidx = torch::randint(int32_min, int32_max, {static_cast<int64_t>(gidx_len)}, options);
  gidx = gidx.repeat({static_cast<int64_t>(num_experts), 1}).contiguous();

  perm = marlin_helper_->MarlinSortGIdx(gidx);

  // 数据检查
  for (size_t e = 0; e < num_experts; e++) {
    for (size_t i = 0; i < 10; i++) {
      EXPECT_TRUE(gidx[e][i].item<int32_t>() == gidx_ref_0[i]);
      EXPECT_TRUE(perm[e][i].item<int32_t>() == perm_ref_0[i]);
    }
  }

  for (size_t e = 0; e < num_experts; e++) {
    for (size_t i = 0; i < 10; i++) {
      EXPECT_TRUE(gidx[e][gidx_len - 10 + i].item<int32_t>() == gidx_ref_1[i]);
      EXPECT_TRUE(perm[e][gidx_len - 10 + i].item<int32_t>() == perm_ref_1[i]);
    }
  }

#endif
}

TEST_F(MarlinUtilsTest, TestMarlinAwqToMarlinZeroPoints) {
#ifdef ENABLE_CUDA
  size_t num_rows = 1024, num_cols = 1024;
  size_t pack_factor = 32 / 4;

  torch::Tensor tensor = torch::arange(static_cast<int64_t>(num_rows * num_cols / pack_factor), torch::kInt32)
                             .to(torch::Device(torch::kCUDA, rank_));
  tensor = tensor.view({static_cast<int64_t>(num_rows), static_cast<int64_t>(num_cols / pack_factor)});

  torch::Tensor zp = marlin_helper_->MarlinAwqToMarlinZeroPoints(tensor, num_rows, num_cols);

  // shape检查
  EXPECT_TRUE(zp.size(0) == 1024);
  EXPECT_TRUE(zp.size(1) == 128);

  // 数据检查
  zp = zp.flatten();
  std::vector<int32_t> zp_ref_0 = {1966171168, 0, 0, 0, 0, 0, 0, 0, -38146904, 0};
  std::vector<int32_t> zp_ref_1 = {-1, 0, -38146904, 286331153, -1, 0, -1, 0, -1, 0};
  for (int i = 0; i < 10; i++) {
    EXPECT_TRUE(zp[i].item<int32_t>() == zp_ref_0[i]);
    EXPECT_TRUE(zp[num_rows * num_cols / pack_factor - 10 + i].item<int32_t>() == zp_ref_1[i]);
  }
#endif
}
