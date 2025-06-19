/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include <endian.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include "logger.h"

namespace ksana_llm {

GGUFFileTensorLoader::GGUFFileTensorLoader(const std::string& file_name, const bool load_bias)
    : BaseFileTensorLoader(file_name, load_bias), gguf_file_(file_name, std::ios::binary | std::ios::ate) {
  // Check if the file name has a ".gguf" extension
  if (file_name_.length() > 5) {
    if (file_name_.substr(file_name_.length() - 5) == ".gguf") {
      // Load the gguf file
      context_ = std::make_shared<GGUFContext>();
      if (!gguf_file_.is_open()) {
        KLLM_THROW(fmt::format("Can't open GGUF file: {}", file_name_));
      }
      file_size_ = gguf_file_.tellg();
      gguf_file_.seekg(0, std::ios::beg);
      LoadGGUFContext();
      if (context_->header.tensor_count > 0) {
        LoadGGUFData();
      }
    }
  }
}

template <typename T>
T GGUFFileTensorLoader::ReadDataFromFile() {
  T value_raw;
  if (!gguf_file_.read(reinterpret_cast<char*>(&value_raw), sizeof(T))) {
    throw std::runtime_error("Failed to read data from file.");
  }
  T value = 0;
  if constexpr (sizeof(T) == 1) {
    value = value_raw;
  } else if constexpr (sizeof(T) == 2) {
    value = le16toh(value_raw);
  } else if constexpr (sizeof(T) == 4) {
    value = le32toh(value_raw);
  } else if constexpr (sizeof(T) == 8) {
    value = le64toh(value_raw);
  }
  return value;
}

std::string GGUFFileTensorLoader::ReadStringFromFile() {
  uint64_t length;
  gguf_file_.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));
  std::string value(length, '\0');
  gguf_file_.read(&value[0], length);
  return value;
}

GGUFFileTensorLoader::~GGUFFileTensorLoader() {
  if (weights_buffer_ != nullptr) {
    delete[] weights_buffer_;
  }
  if (gguf_file_.is_open()) {
    gguf_file_.close();
  }
}

DataType GGUFFileTensorLoader::ConvertGGMLTypeToDataType(uint32_t ggml_type) {
  switch (ggml_type) {
    case GGMLType::GGML_TYPE_F32:  // GGML_TYPE_F32
      return TYPE_FP32;
    case GGMLType::GGML_TYPE_F16:  // GGML_TYPE_F16
      return TYPE_FP16;
    case GGMLType::GGML_TYPE_I8:  // GGML_TYPE_I8
      return TYPE_INT8;
    case GGMLType::GGML_TYPE_I16:  // GGML_TYPE_I16
      return TYPE_INT16;
    case GGMLType::GGML_TYPE_I32:  // GGML_TYPE_I32
      return TYPE_INT32;
    case GGMLType::GGML_TYPE_I64:  // GGML_TYPE_I64
      return TYPE_INT64;
    case GGMLType::GGML_TYPE_F64:  // GGML_TYPE_F64
      return TYPE_FP64;
    case GGMLType::GGML_TYPE_BF16:  // GGML_TYPE_Q4_0
      return TYPE_BF16;
    default:
      return TYPE_INVALID;
  }
}

DataType GGUFFileTensorLoader::ConverGGUFModelFileTypeToDataType(uint32_t gguf_model_file_type) {
  switch (gguf_model_file_type) {
    case GGUFModelFileType::LLAMA_FTYPE_ALL_F32:
      return DataType::TYPE_FP32;
    case GGUFModelFileType::LLAMA_FTYPE_MOSTLY_F16:
      return DataType::TYPE_FP16;
    case GGUFModelFileType::LLAMA_FTYPE_MOSTLY_BF16:
      return DataType::TYPE_BF16;
    default:
      return TYPE_INVALID;
  }
}

std::any GGUFFileTensorLoader::ReadGGUFMetadataValue(GGUFMetaValueType type) {
  switch (type) {
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_UINT8: {
      uint8_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(uint8_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_INT8: {
      int8_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(int8_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_UINT16: {
      uint16_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(uint16_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_INT16: {
      int16_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(int16_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_UINT32: {
      uint32_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_INT32: {
      int32_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_UINT64: {
      uint64_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_INT64: {
      int64_t value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(int64_t));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_FLOAT32: {
      float value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(float));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_FLOAT64: {
      double value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(double));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_BOOL: {
      bool value;
      gguf_file_.read(reinterpret_cast<char*>(&value), sizeof(bool));
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_STRING: {
      uint64_t length;
      gguf_file_.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));
      std::string value(length, '\0');
      gguf_file_.read(&value[0], length);
      return value;
    }
    case GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_ARRAY: {
      uint32_t elem_type_raw;
      gguf_file_.read(reinterpret_cast<char*>(&elem_type_raw), sizeof(uint32_t));
      GGUFMetaValueType elem_type = static_cast<GGUFMetaValueType>(elem_type_raw);

      uint64_t length;
      gguf_file_.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));

      std::vector<std::any> array_values;
      for (uint64_t i = 0; i < length; ++i) {
        array_values.push_back(ReadGGUFMetadataValue(elem_type));
      }
      return array_values;
    }
    default:
      throw std::runtime_error("Unsupported type encountered");
  }
}

void GGUFFileTensorLoader::LoadGGUFContext() {
  gguf_file_.read(reinterpret_cast<char*>(&context_->header.magic), sizeof(uint32_t));
  if (context_->header.magic != GGUF_MAGIC) {
    throw std::runtime_error("Invalid GGUF magic number.");
  }

  gguf_file_.read(reinterpret_cast<char*>(&context_->header.version), sizeof(uint32_t));

  if (context_->header.version != GGUF_VERSION) {
    throw std::runtime_error("Unsupported GGUF version.");
  }

  gguf_file_.read(reinterpret_cast<char*>(&context_->header.tensor_count), sizeof(uint64_t));
  gguf_file_.read(reinterpret_cast<char*>(&context_->header.metadata_kv_count), sizeof(uint64_t));

  for (uint64_t i = 0; i < context_->header.metadata_kv_count; ++i) {
    GGUFMetaValue meta_data;
    uint64_t key_length;
    gguf_file_.read(reinterpret_cast<char*>(&key_length), sizeof(uint64_t));

    if (key_length > MAX_STRING_LENGTH) {
      throw std::runtime_error("Invalid key length in metadata.");
    }

    std::string key(key_length, '\0');
    gguf_file_.read(reinterpret_cast<char*>(&key[0]), key_length);

    uint32_t value_type_int;
    gguf_file_.read(reinterpret_cast<char*>(&value_type_int), sizeof(uint32_t));
    meta_data.type = static_cast<GGUFMetaValueType>(value_type_int);
    meta_data.value = ReadGGUFMetadataValue(meta_data.type);
    context_->metadata_map[key] = meta_data;
  }

  for (uint64_t i = 0; i < context_->header.tensor_count; ++i) {
    GGUFTensorInfo tensor_info;
    tensor_info.name = ReadStringFromFile();
    tensor_info.n_dims = ReadDataFromFile<uint32_t>();
    for (uint32_t j = 0; j < tensor_info.n_dims; ++j) {
      tensor_info.dims.push_back(ReadDataFromFile<uint64_t>());
    }

    uint32_t data_type_int = ReadDataFromFile<uint32_t>();
    tensor_info.data_type = ConvertGGMLTypeToDataType(data_type_int);
    tensor_info.offset = ReadDataFromFile<uint64_t>();
    size_t tensor_data_size = 1;
    for (uint32_t dim : tensor_info.dims) {
      if (dim == 0 || tensor_data_size > SIZE_MAX / dim) {
        throw std::overflow_error("Tensor size calculation overflow.");
      }
      tensor_data_size *= dim;
    }
    tensor_info.size = tensor_data_size * GetTypeSize(tensor_info.data_type);
    context_->tensor_info_map[tensor_info.name] = tensor_info;
    tensor_name_list_.push_back(tensor_info.name);
  }

  size_t alignment = context_->metadata_map.count("general.alignment")
                         ? std::any_cast<uint32_t>(context_->metadata_map["general.alignment"].value)
                         : GGUF_ALIGNMENT;

  auto offset = gguf_file_.tellg();
  size_t offset_pad = offset % alignment;
  if (offset_pad != 0) {
    offset += alignment - offset_pad;
  }
  context_->offset = offset;
  context_->alignment = alignment;
}

void GGUFFileTensorLoader::LoadGGUFData() {
  size_t data_size = file_size_ - context_->offset;
  try {
    weights_buffer_ = new char[data_size];
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("Failed to allocate memory for weights_buffer_.");
  }

  gguf_file_.seekg(context_->offset, std::ios::beg);
  gguf_file_.read(weights_buffer_, data_size);

  for (const auto& item : context_->tensor_info_map) {
    const GGUFTensorInfo& tensor_info = item.second;
    if (!load_bias_ && tensor_info.name.find(".bias") != std::string::npos) {
      continue;
    }
    tensor_ptr_map_[tensor_info.name] = weights_buffer_ + tensor_info.offset;
  }
}

const std::vector<std::string>& GGUFFileTensorLoader::GetTensorNameList() { return tensor_name_list_; }

std::tuple<void*, size_t> GGUFFileTensorLoader::GetTensor(const std::string& tensor_name) {
  if (!context_->tensor_info_map.count(tensor_name)) {
    return std::make_tuple(nullptr, 0);
  }
  return std::make_tuple(tensor_ptr_map_[tensor_name], context_->tensor_info_map[tensor_name].size);
}

void GGUFFileTensorLoader::SetTensor(const std::string& tensor_name, torch::Tensor tensor) {
  if (!context_->tensor_info_map.count(tensor_name)) {
    return;
  }
  tensor_map_[tensor_name] = std::move(tensor);
  tensor_ptr_map_[tensor_name] = tensor_map_[tensor_name].data_ptr();
}

DataType GGUFFileTensorLoader::GetTensorDataType(const std::string& tensor_name) {
  if (!context_->tensor_info_map.count(tensor_name)) {
    return TYPE_INVALID;
  }
  return context_->tensor_info_map[tensor_name].data_type;
}

std::string GGUFFileTensorLoader::GetTensorFileName() { return file_name_; }

void GGUFFileTensorLoader::InitTokenizer(const std::string& model_dir_path) {
  std::string tokenizer_model;
  std::string tokenizer_pre;
}

std::vector<size_t> GGUFFileTensorLoader::GetTensorShape(const std::string& tensor_name) {
  if (!context_->tensor_info_map.count(tensor_name)) {
    return {};
  }

  // GGUF dimension order is reversed compared to Python
  // https://github.com/ggerganov/ggml/issues/500
  std::vector<size_t> dims = context_->tensor_info_map[tensor_name].dims;
  std::reverse(dims.begin(), dims.end());
  return dims;
}

std::string GGUFFileTensorLoader::ConvertFormatToRegex() {
  std::string regex_str;
  const std::string regex_special_chars = ".^$|()[]{}*+?\\";
  const std::string format = "{:s}-{:05d}-of-{:05d}.gguf";

  for (size_t i = 0; i < format.size(); ++i) {
    if (format[i] == '{') {
      size_t end_pos = format.find('}', i);
      if (end_pos != std::string::npos) {
        std::string placeholder = format.substr(i, end_pos - i + 1);
        if (placeholder == "{:s}") {
          regex_str += ".*";
        } else if (placeholder == "{:05d}") {
          regex_str += "\\d{5}";
        } else {
          regex_str += ".*";
        }
        i = end_pos;
      } else {
        if (regex_special_chars.find(format[i]) != std::string::npos) {
          regex_str += '\\';
        }
        regex_str += format[i];
      }
    } else {
      if (regex_special_chars.find(format[i]) != std::string::npos) {
        regex_str += '\\';
      }
      regex_str += format[i];
    }
  }
  regex_str += '$';
  return regex_str;
}

std::vector<std::string> GGUFFileTensorLoader::FindModelFiles(const std::string& model_dir_path) {
  std::string shard_regex_str = ConvertFormatToRegex();
  std::regex shard_regex(shard_regex_str, std::regex_constants::ECMAScript | std::regex_constants::optimize);

  std::vector<std::string> gguf_files;
  std::vector<std::string> shard_files;

  for (const auto& entry : std::filesystem::directory_iterator(model_dir_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const auto& path = entry.path();
    if (path.extension() == ".gguf") {
      gguf_files.push_back(path.string());
      const auto& filename = path.filename().string();
      if (std::regex_match(filename, shard_regex)) {
        shard_files.push_back(path.string());
      }
    }
  }

  if (gguf_files.empty()) {
    std::string error_msg = "No .gguf files found in model directory: " + model_dir_path;
    KLLM_LOG_ERROR << error_msg << std::endl;
    return {};
  }

  if (!shard_files.empty()) {
    std::sort(shard_files.begin(), shard_files.end());
    return shard_files;
  }

  return gguf_files;
}

}  // namespace ksana_llm