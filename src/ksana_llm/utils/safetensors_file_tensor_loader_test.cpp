// Copyright 2024 Tencent Inc.  All rights reserved.
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"
#include "gtest/gtest.h"
#include "logger.h"

using namespace ksana_llm;
#ifdef ENABLE_CUDA
class SafetensorsLoaderTest : public testing::Test {};
// Test case for SafeTensorsLoader
TEST_F(SafetensorsLoaderTest, SafetensorsLoaderBiasTest) {
  const std::string file_name = "/model/qwen1.5-hf/0.5B-Chat/model.safetensors";
  SafeTensorsLoader safetensors_loader(file_name, true);
  EXPECT_EQ(safetensors_loader.GetTensorFileName(), file_name);

  bool exist_bias = false;
  const auto& tensor_name_list_ = safetensors_loader.GetTensorNameList();
  for (const auto& tensor_name : tensor_name_list_) {
    const auto& [ptr, size] = safetensors_loader.GetTensor(tensor_name);
    const auto& dtype = safetensors_loader.GetTensorDataType(tensor_name);
    const auto& shape = safetensors_loader.GetTensorShape(tensor_name);

    if (tensor_name.find(".bias") != std::string::npos) {
      exist_bias = true;
    }
    EXPECT_NE(ptr, nullptr);
    EXPECT_NE(size, 0);
    EXPECT_NE(dtype, TYPE_INVALID);
  }
  EXPECT_TRUE(exist_bias);

  const auto& [ptr, shape] = safetensors_loader.GetTensor("invalid_name");
  EXPECT_EQ(ptr, nullptr);
  const auto& dtype = safetensors_loader.GetTensorDataType("invalid_name");
  EXPECT_EQ(dtype, TYPE_INVALID);
}

TEST_F(SafetensorsLoaderTest, SafetensorsLoaderNoBiasTest) {
  const std::string file_name = "/model/qwen1.5-hf/0.5B-Chat/model.safetensors";
  SafeTensorsLoader safetensors_loader(file_name, false);
  EXPECT_EQ(safetensors_loader.GetTensorFileName(), file_name);

  const auto& tensor_name_list_ = safetensors_loader.GetTensorNameList();
  for (const auto& tensor_name : tensor_name_list_) {
    const auto& [ptr, size] = safetensors_loader.GetTensor(tensor_name);
    const auto& dtype = safetensors_loader.GetTensorDataType(tensor_name);
    const auto& shape = safetensors_loader.GetTensorShape(tensor_name);

    EXPECT_EQ(tensor_name.find(".bias"), std::string::npos);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NE(size, 0);
    EXPECT_NE(dtype, TYPE_INVALID);
  }
}
#endif