// Copyright 2024 Tencent Inc.  All rights reserved.
#include "pytorch_file_tensor_loader.h"
#include "logger.h"
#include "test.h"

namespace ksana_llm {

// Test case for PytorchFileTensorLoader
TEST(PytorchFileTensorLoaderTest, PytorchFileTensorLoaderTest) {
  PytorchFileTensorLoader loader("/model/llama-hf/7B/pytorch_model-00002-of-00002.bin", true);
  auto tensor_name_list = loader.GetTensorNameList();

  // Iterate through the tensor names and check if the tensor can be loaded
  for (auto i : tensor_name_list) {
    EXPECT_NE(std::get<0>(loader.GetTensor(i)), nullptr);  // Expect the tensor to be loaded successfully
  }
}

}  // namespace ksana_llm
