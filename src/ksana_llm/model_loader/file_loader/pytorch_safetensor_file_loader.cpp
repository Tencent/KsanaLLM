/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/file_loader/pytorch_safetensor_file_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <unordered_set>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

PytorchSafetensorFileLoader::PytorchSafetensorFileLoader(const std::string& filename) : filename_(filename) {
  const int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << filename;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << filename;
    close(fd);
    return;
  }
  file_size_ = sb.st_size;

  mmap_ptr_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mmap_ptr_ == MAP_FAILED) {
    KLLM_LOG_FATAL << "Can't open safetensors file: " << filename;
    close(fd);
    return;
  }
  close(fd);
}

PytorchSafetensorFileLoader::~PytorchSafetensorFileLoader() {
  if (mmap_ptr_ != nullptr) {
    munmap(mmap_ptr_, file_size_);
    mmap_ptr_ = nullptr;
  }
}

DataType GetTensorDataType(const std::string& safetensor_dtype) {
  const std::map<std::string, DataType> type_map = {
      {"F16", TYPE_FP16},         {"F32", TYPE_FP32},  {"BF16", TYPE_BF16}, {"I32", TYPE_INT32},
      {"F8_E4M3", TYPE_FP8_E4M3}, {"UI8", TYPE_UINT8}, {"U8", TYPE_UINT8},  {"I8", TYPE_INT8}};
  return type_map.at(safetensor_dtype);
}

Status PytorchSafetensorFileLoader::LoadSafetensorTensorDict() {
  if (!tensor_dict_.empty()) {
    return Status();
  }

  char* data_ptr = reinterpret_cast<char*>(mmap_ptr_);

  // get the tensor list(string)
  const size_t header_size = *reinterpret_cast<const size_t*>(data_ptr);
  data_ptr += sizeof(header_size);

  const std::string_view tensor_dict_str(reinterpret_cast<const char*>(data_ptr), header_size);
  data_ptr += header_size;
  KLLM_LOG_DEBUG << fmt::format("Safetensors file {} Header = {}", filename_, tensor_dict_str);

  // Parsing JSON to retrieve tensor information.
  tensor_dict_ = json::parse(tensor_dict_str);
  return Status();
}

Status PytorchSafetensorFileLoader::LoadWeightNames(std::vector<std::string>& weight_names) {
  const Status status = LoadSafetensorTensorDict();
  if (!status.OK()) {
    return status;
  }

  weight_names.clear();
  weight_names.reserve(tensor_dict_.size());
  for (const auto& [weight_name, tensor_data] : tensor_dict_.items()) {
    if (!tensor_data.contains("data_offsets")) {
      continue;
    }
    weight_names.emplace_back(weight_name);
  }

  return Status();
}

Status PytorchSafetensorFileLoader::LoadModelWeights(const std::vector<std::string>& weight_names,
                                                     std::unordered_map<std::string, Tensor>& result) {
  const Status status = LoadSafetensorTensorDict();
  if (!status.OK()) {
    return status;
  }

  char* data_ptr = reinterpret_cast<char*>(mmap_ptr_);
  const size_t header_size = *reinterpret_cast<const size_t*>(data_ptr);
  data_ptr += sizeof(header_size) + header_size;

  result.reserve(tensor_dict_.size());
  const std::unordered_set<std::string> weight_name_set(weight_names.begin(), weight_names.end());
  for (const auto& [weight_name, tensor_data] : tensor_dict_.items()) {
    if (weight_name_set.find(weight_name) == weight_name_set.end()) {
      KLLM_LOG_DEBUG << "Skip weight tensor name " << weight_name;
      continue;
    }

    std::vector<size_t> tensor_shape = tensor_data["shape"].get<decltype(tensor_shape)>();
    if (tensor_shape.size() == 0) {
      // 如果Tensor只有一个值，shape有可能是空()，强制修改为(1)
      tensor_shape = {1};
    }
    KLLM_LOG_DEBUG << FormatStr("SafeTensors Loader: weight_name:%s, shape:%s", weight_name.c_str(),
                                Vector2Str(tensor_shape).c_str());

    const size_t tensor_beg_index = tensor_data["data_offsets"][0];
    const DataType tensor_dtype = GetTensorDataType(tensor_data["dtype"]);
    result[weight_name] = Tensor(MemoryLocation::LOCATION_HOST, tensor_dtype, tensor_shape, -1,
                                 reinterpret_cast<void*>(data_ptr) + tensor_beg_index);
  }

  return Status();
}

}  // namespace ksana_llm
