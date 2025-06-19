/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "ksana_llm/utils/base_file_tensor_loader.h"

namespace ksana_llm {
// Create a MockSafeTensorsLoader class to simulate the behavior of SafeTensorsLoader
class MockSafeTensorsLoader : public BaseFileTensorLoader {
 public:
  explicit MockSafeTensorsLoader(const std::string& file_name, const bool load_bias)
      : BaseFileTensorLoader(file_name, load_bias) {}

  ~MockSafeTensorsLoader() {
    for (auto& [name, ptr] : tensor_ptr_map_) {
      if (ptr) {
        free(ptr);
      }
    }
  }

  const std::vector<std::string>& GetTensorNameList() override { return tensor_name_list_; }

  std::tuple<void*, size_t> GetTensor(const std::string& tensor_name) override {
    if (tensor_ptr_map_.find(tensor_name) == tensor_ptr_map_.end() ||
        tensor_size_map_.find(tensor_name) == tensor_size_map_.end()) {
      return std::make_tuple(nullptr, 0);
    }
    return std::make_tuple(tensor_ptr_map_[tensor_name], tensor_size_map_[tensor_name]);
  }

  void SetTensor(const std::string& tensor_name, torch::Tensor tensor) override {
    KLLM_THROW(fmt::format("SetTensor not implement {}.", tensor_name));
  }

  DataType GetTensorDataType(const std::string& tensor_name) override {
    if (tensor_data_type_map_.find(tensor_name) == tensor_data_type_map_.end()) {
      return TYPE_INVALID;
    }
    return tensor_data_type_map_[tensor_name];
  }

  std::string GetTensorFileName() override { return file_name_; }

  std::vector<size_t> GetTensorShape(const std::string& tensor_name) override {
    if (tensor_shape_map_.find(tensor_name) == tensor_shape_map_.end()) {
      return {};
    }
    return tensor_shape_map_[tensor_name];
  }

 private:
  virtual void InitMockData() = 0;

  virtual void CreateMockTensor(const std::string& tensor_name, const std::vector<size_t>& shape, DataType data_type,
                                size_t expert_idx) = 0;

  std::unordered_map<std::string, void*> tensor_ptr_map_;
  std::unordered_map<std::string, size_t> tensor_size_map_;
  std::unordered_map<std::string, DataType> tensor_data_type_map_;
  std::unordered_map<std::string, std::vector<size_t>> tensor_shape_map_;
};

}  // namespace ksana_llm