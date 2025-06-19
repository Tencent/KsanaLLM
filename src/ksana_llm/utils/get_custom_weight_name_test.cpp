/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/get_custom_weight_name.h"

#include "gtest/gtest.h"

// Namespace alias for convenience
namespace ksana = ksana_llm;

class GetCustomWeightNameTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Test Getting weight_map from ksana_llm
TEST_F(GetCustomWeightNameTest, GetWeightMapPathTest) {
  std::string model_path = "/not/exsit/path";
  std::string model_type = "hunyuan";
  ksana::ModelFileFormat model_file_format = ksana::ModelFileFormat::SAFETENSORS;
  std::string weight_path = ksana::GetWeightMapPath(model_path, model_type, model_file_format);
  EXPECT_NE(weight_path, "");
}
