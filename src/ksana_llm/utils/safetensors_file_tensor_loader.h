// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include "base_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {
// Define a class named PytorchFileTensorLoader that inherits from BaseFileTensorLoader
class SafeTensorsLoader : public BaseFileTensorLoader {
 public:
  // Constructor that takes a file name as input
  explicit SafeTensorsLoader(const std::string& file_name, const bool load_bias);

  ~SafeTensorsLoader();

  // Get the list of tensor names
  const std::vector<std::string>& GetTensorNameList() { return tensor_name_list_; }

  // Get a tensor by its name
  std::tuple<void*, size_t> GetTensor(const std::string& tensor_name);

  void SetTensor(const std::string& tensor_name, torch::Tensor tensor) {
    KLLM_THROW(fmt::format("SetTensor not implement {}.", tensor_name));
  }

  DataType GetTensorDataType(const std::string& tensor_name);

  std::string GetTensorFileName();

  std::vector<std::size_t> GetTensorShape(const std::string& tensor_name);

 private:
  // Load the PyTorch binary file
  void LoadSafeTensors();

  DataType ConvertDtypeToDataType(const std::string& safetensors_dtype);

 private:
  // Use unordered_map to store the tensor names and their data ptr
  std::unordered_map<std::string, void*> tensor_ptr_map_;
  std::unordered_map<std::string, size_t> tensor_size_map_;
  std::unordered_map<std::string, DataType> tensor_data_type_map_;
  std::unordered_map<std::string, std::vector<size_t>> tensor_shape_map_;

  void* mmap_ptr_ = nullptr;
  size_t file_size_;
};

}  // namespace ksana_llm
