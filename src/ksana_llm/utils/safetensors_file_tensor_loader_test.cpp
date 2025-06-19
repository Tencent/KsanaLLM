// Copyright 2024 Tencent Inc.  All rights reserved.
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"
#include "gtest/gtest.h"
#include "logger.h"

using namespace ksana_llm;
#ifdef ENABLE_CUDA
class SafetensorsLoaderTest : public testing::Test {
 protected:
  bool load_bias_;
  std::vector<std::string> tensor_name_list_;

  bool CheckBiasName() {
    KLLM_LOG_INFO << "load_bias: " << load_bias_;
    SafeTensorsLoader safetensors_loader("/model/qwen1.5-hf/0.5B-Chat/model.safetensors", load_bias_);
    tensor_name_list_ = safetensors_loader.GetTensorNameList();
    for (const auto& tensor_name : tensor_name_list_) {
      if (tensor_name.find(".bias") != std::string::npos) {
        return true;
      }
    }
    return false;
  }
};
// Test case for SafeTensorsLoader
TEST_F(SafetensorsLoaderTest, SafetensorsLoaderBiasTest) {
  // Load bias
  load_bias_ = true;
  EXPECT_EQ(CheckBiasName(), true);
}

TEST_F(SafetensorsLoaderTest, SafetensorsLoaderNoBiasTest) {
  // Load no bias
  load_bias_ = false;
  EXPECT_EQ(CheckBiasName(), false);
}
#endif