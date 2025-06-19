/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Class for saving tensors to safetensors format files
class SafetensorsFileSaver {
 public:
  explicit SafetensorsFileSaver(const std::string& base_file_name, size_t rank, std::shared_ptr<Context> context,
                                size_t max_file_size_bytes = 1024 * 1024 * 1024);

  // Save tensors from weights_map to safetensors files
  Status SaveTensors(const std::unordered_map<std::string, Tensor>& weights_map);

 private:
  // Convert DataType to safetensors dtype string
  std::string ConvertDataTypeToSafetensorsDtype(DataType data_type);

  // Group tensors into files based on max file size
  std::vector<std::vector<std::string>> GroupTensorsIntoFiles(
      const std::unordered_map<std::string, Tensor>& weights_map);

  // Save a group of tensors to a single safetensors file
  Status SaveTensorGroup(const std::vector<std::string>& tensor_names,
                         const std::unordered_map<std::string, Tensor>& weights_map, const std::string& file_name);

  std::string base_file_name_;
  size_t max_file_size_bytes_;

  size_t rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
};

}  // namespace ksana_llm
