/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "safetensors_file_tensor_loader.h"
#include <nlohmann/json.hpp>
#include "logger.h"

using json = nlohmann::json;

namespace ksana_llm {
// Constructor of SafeTensorsLoader that takes a file name as input
SafeTensorsLoader::SafeTensorsLoader(const std::string& file_name, const bool load_bias)
    : BaseFileTensorLoader(file_name, load_bias) {
  // Check if the file name has a ".bin" extension
  if (file_name_.length() > 12) {
    if (file_name_.substr(file_name_.length() - 12) == ".safetensors") {
      // Load the PyTorch binary file
      LoadSafeTensors();
    }
  }
}

SafeTensorsLoader::~SafeTensorsLoader() { delete[] weights_buffer_; }

DataType SafeTensorsLoader::ConvertDtypeToDataType(const std::string& safetensors_dtype) {
  if (safetensors_dtype == "F16") {
    return TYPE_FP16;
  } else if (safetensors_dtype == "F32") {
    return TYPE_FP32;
  } else if (safetensors_dtype == "BF16") {
    return TYPE_BF16;
  } else if (safetensors_dtype == "I32") {
    return TYPE_INT32;
  } else if (safetensors_dtype == "F8_E4M3") {
    return TYPE_FP8_E4M3;
  } else if (safetensors_dtype == "UI8") {
    return TYPE_UINT8;
  }
  return TYPE_INVALID;
}

// Function to load the SafeTensors binary file
void SafeTensorsLoader::LoadSafeTensors() {
  std::ifstream safetensors_file(file_name_, std::ios::binary | std::ios::ate);
  if (!safetensors_file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Can't open safetensors file: {}", file_name_);
  }
  int64_t file_size = safetensors_file.tellg();
  if (file_size == -1) {
    KLLM_LOG_ERROR << fmt::format("Invalid safetensors file size: -1, filename: {}", file_name_);
  }
  safetensors_file.seekg(0, std::ios::beg);

  // get the tensor list(string)
  size_t header_size;
  safetensors_file.read(reinterpret_cast<char*>(&header_size), sizeof(size_t));
  std::string tensor_dict_str;
  tensor_dict_str.resize(header_size);
  safetensors_file.read(&tensor_dict_str[0], header_size);
  KLLM_LOG_DEBUG << fmt::format("Safetensors file {} Header = {}", file_name_, tensor_dict_str);

  size_t data_size = file_size - header_size - sizeof(size_t);
  weights_buffer_ = new char[data_size];
  safetensors_file.read(weights_buffer_, data_size);

  // Parsing JSON to retrieve tensor information.
  json tensor_dict = json::parse(tensor_dict_str);
  for (const auto& tensor_iter : tensor_dict.items()) {
    // tensor name
    const std::string& tensor_name = tensor_iter.key();
    if (!load_bias_ && tensor_name.find(".bias") != std::string::npos) {
      continue;
    }
    tensor_name_list_.emplace_back(tensor_name);
    json tensor_data = tensor_iter.value();
    if (!tensor_data.contains("data_offsets")) {
      continue;
    }
    // tensor ptr
    size_t tensor_begin_index = tensor_data["data_offsets"][0];
    size_t tensor_end_index = tensor_data["data_offsets"][1];

    std::string tensor_dtype_str = tensor_data["dtype"];
    tensor_data_type_map_[tensor_name] = ConvertDtypeToDataType(tensor_dtype_str);
    KLLM_LOG_DEBUG << fmt::format("SafeTensors Loader: tensor_name = {}, dtype = {}", tensor_name, tensor_dtype_str);
    tensor_shape_map_[tensor_name] = {};
    for (size_t dim : tensor_data["shape"]) {
      tensor_shape_map_[tensor_name].emplace_back(dim);
    }
    tensor_offset_map_[tensor_name] = tensor_begin_index;
    tensor_size_map_[tensor_name] = tensor_end_index - tensor_begin_index;

    safetensors_file.seekg(tensor_offset_map_["base_index"], std::ios::cur);
    safetensors_file.read(weights_buffer_ + tensor_offset_map_[tensor_name], tensor_size_map_[tensor_name]);
    tensor_ptr_map_[tensor_name] = weights_buffer_ + tensor_offset_map_[tensor_name];
  }
  safetensors_file.close();
}

// Function to get a tensor by its name
std::tuple<void*, size_t> SafeTensorsLoader::GetTensor(const std::string& tensor_name) {
  // Check if the tensor name exists in the index map
  if (!tensor_ptr_map_.count(tensor_name) || !tensor_offset_map_.count(tensor_name) ||
      !tensor_size_map_.count(tensor_name)) {
    return std::make_tuple(nullptr, 0);
  }
  return std::make_tuple(tensor_ptr_map_[tensor_name], tensor_size_map_[tensor_name]);
}

DataType SafeTensorsLoader::GetTensorDataType(const std::string& tensor_name) {
  if (!tensor_data_type_map_.count(tensor_name)) {
    return TYPE_INVALID;
  }
  return tensor_data_type_map_[tensor_name];
}

std::string SafeTensorsLoader::GetTensorFileName() { return file_name_; }

std::vector<size_t> SafeTensorsLoader::GetTensorShape(const std::string& tensor_name) {
  if (!tensor_data_type_map_.count(tensor_name)) {
    return {};
  }
  return tensor_shape_map_[tensor_name];
}

}  // namespace ksana_llm
