/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "safetensors_file_tensor_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>

#include <nlohmann/json.hpp>
#include "logger.h"

using json = nlohmann::json;

namespace ksana_llm {
// Constructor of SafeTensorsLoader that takes a file name as input
SafeTensorsLoader::SafeTensorsLoader(const std::string& file_name, const bool load_bias)
    : BaseFileTensorLoader(file_name, load_bias) {
  if (std::filesystem::path(file_name).extension().string() != ".safetensors") {
    return;
  }
  LoadSafeTensors();
}

SafeTensorsLoader::~SafeTensorsLoader() {
  if (mmap_ptr_ != nullptr) {
    munmap(mmap_ptr_, file_size_);
    mmap_ptr_ = nullptr;
  }
}

DataType SafeTensorsLoader::ConvertDtypeToDataType(const std::string& safetensors_dtype) {
  const std::map<std::string, DataType> type_map = {{"F16", TYPE_FP16},         {"F32", TYPE_FP32},
                                                    {"BF16", TYPE_BF16},        {"I32", TYPE_INT32},
                                                    {"F8_E4M3", TYPE_FP8_E4M3}, {"UI8", TYPE_UINT8}};
  return type_map.at(safetensors_dtype);
}

// Function to load the SafeTensors binary file
void SafeTensorsLoader::LoadSafeTensors() {
  const int fd = open(file_name_.c_str(), O_RDONLY);
  if (fd < 0) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << file_name_;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << file_name_;
    close(fd);
    return;
  }
  file_size_ = sb.st_size;

  mmap_ptr_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mmap_ptr_ == MAP_FAILED) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << file_name_;
    close(fd);
    return;
  }
  close(fd);

  char* data_ptr = reinterpret_cast<char*>(mmap_ptr_);

  // get the tensor list(string)
  const size_t header_size = *reinterpret_cast<const size_t*>(data_ptr);
  data_ptr += sizeof(header_size);

  const std::string_view tensor_dict_str(reinterpret_cast<const char*>(data_ptr), header_size);
  data_ptr += header_size;
  KLLM_LOG_DEBUG << fmt::format("Safetensors file {} Header = {}", file_name_, tensor_dict_str);

  // Parsing JSON to retrieve tensor information.
  const json& tensor_dict = json::parse(tensor_dict_str);
  KLLM_LOG_INFO << "json parse done";

  tensor_name_list_.reserve(tensor_dict.size());
  for (const auto& [tensor_name, tensor_data] : tensor_dict.items()) {
    if (!load_bias_ && tensor_name.find(".bias") != std::string::npos) {
      continue;
    }
    if (!tensor_data.contains("data_offsets")) {
      continue;
    }
    tensor_name_list_.emplace_back(tensor_name);

    const std::string& tensor_dtype_str = tensor_data["dtype"];
    tensor_data_type_map_[tensor_name] = ConvertDtypeToDataType(tensor_dtype_str);
    KLLM_LOG_DEBUG << fmt::format("SafeTensors Loader: tensor_name = {}, dtype = {}", tensor_name, tensor_dtype_str);
    tensor_shape_map_[tensor_name] = tensor_data["shape"].get<decltype(tensor_shape_map_)::mapped_type>();

    const size_t tensor_begin_index = tensor_data["data_offsets"][0];
    const size_t tensor_end_index = tensor_data["data_offsets"][1];
    tensor_size_map_[tensor_name] = tensor_end_index - tensor_begin_index;
    tensor_ptr_map_[tensor_name] = reinterpret_cast<void*>(data_ptr) + tensor_begin_index;
  }
}

// Function to get a tensor by its name
std::tuple<void*, size_t> SafeTensorsLoader::GetTensor(const std::string& tensor_name) {
  // Check if the tensor name exists in the index map
  const auto ptr_it = tensor_ptr_map_.find(tensor_name);
  if (ptr_it == tensor_ptr_map_.end()) {
    return std::make_tuple(nullptr, 0);
  }
  return std::make_tuple(ptr_it->second, tensor_size_map_[tensor_name]);
}

DataType SafeTensorsLoader::GetTensorDataType(const std::string& tensor_name) {
  const auto it = tensor_data_type_map_.find(tensor_name);
  if (it == tensor_data_type_map_.end()) {
    return TYPE_INVALID;
  }
  return it->second;
}

std::string SafeTensorsLoader::GetTensorFileName() { return file_name_; }

std::vector<size_t> SafeTensorsLoader::GetTensorShape(const std::string& tensor_name) {
  const auto it = tensor_shape_map_.find(tensor_name);
  if (it == tensor_shape_map_.end()) {
    return {};
  }
  return it->second;
}

}  // namespace ksana_llm
