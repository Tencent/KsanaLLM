/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

class FusedMoeTest : public testing::Test {};

int RunPythonScript() {
  std::string cmd =
      "python fused_moe.py "
      "--output_dir ./ "
      "--group_n 128 "
      "--group_k 128 "
      "--BLOCK_SIZE_M 64 "
      "--BLOCK_SIZE_N 128 "
      "--BLOCK_SIZE_K 128 "
      "--GROUP_SIZE_M 1 "
      "--m 32 "
      "--MUL_ROUTED_WEIGHT False "
      "--top_k 8 "
      "--compute_type FP16 "
      "--use_fp8_w8a8 True "
      "--use_int8_w8a16 False "
      "--tune "
      "--deep_tune "
      "--model_path /model/DeepSeek-R1-17832-fix-mtp  "
      "--even_Ks True "
      "--tp_size 8";
  int ret = std::system(cmd.c_str());
  if (ret != 0) {
    // run python script failed
    return ret;
  }
  std::ifstream file("best_config.json");
  if (!file.is_open()) {
    // Failed to open the file
    return -1;
  }
  nlohmann::json data;
  file >> data;
  EXPECT_EQ(16, data["32"]["BLOCK_SIZE_M"]);
  EXPECT_EQ(128, data["32"]["BLOCK_SIZE_N"]);
  EXPECT_EQ(128, data["32"]["BLOCK_SIZE_K"]);
  EXPECT_EQ(1, data["32"]["GROUP_SIZE_M"]);
  EXPECT_EQ(4, data["32"]["num_warps"]);
  EXPECT_EQ(3, data["32"]["num_stages"]);
  return 0;
}

TEST_F(FusedMoeTest, Test) {
  int ret = RunPythonScript();
  EXPECT_EQ(0, ret);
}
