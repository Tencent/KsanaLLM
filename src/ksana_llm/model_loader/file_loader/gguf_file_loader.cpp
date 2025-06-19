/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/file_loader/gguf_file_loader.h"

#include <filesystem>
#include <regex>
#include <unordered_set>

#include "ksana_llm/utils/gguf_file_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

GGUFFileLoader::GGUFFileLoader(const std::string& filename) : filename_(filename) {}

GGUFFileLoader::~GGUFFileLoader() {
  if (gguf_file_.is_open()) {
    gguf_file_.close();
    loaded_ = false;
  }
}

Status ReadGGUFMetadataValue(std::ifstream& gguf_file, NewGGUFMetaValueType type, std::any& any_value) {
  switch (type) {
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_UINT8: {
      uint8_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(uint8_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_INT8: {
      int8_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(int8_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_UINT16: {
      uint16_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(uint16_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_INT16: {
      int16_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(int16_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_UINT32: {
      uint32_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_INT32: {
      int32_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_UINT64: {
      uint64_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_INT64: {
      int64_t value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(int64_t));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_FLOAT32: {
      float value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(float));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_FLOAT64: {
      double value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(double));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_BOOL: {
      bool value;
      gguf_file.read(reinterpret_cast<char*>(&value), sizeof(bool));
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_STRING: {
      uint64_t length;
      gguf_file.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));
      std::string value(length, '\0');
      gguf_file.read(&value[0], length);
      any_value = value;
      break;
    }
    case NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_ARRAY: {
      uint32_t elem_type_raw;
      gguf_file.read(reinterpret_cast<char*>(&elem_type_raw), sizeof(uint32_t));
      NewGGUFMetaValueType elem_type = static_cast<NewGGUFMetaValueType>(elem_type_raw);

      uint64_t length;
      gguf_file.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));

      std::vector<std::any> array_values;
      for (uint64_t i = 0; i < length; ++i) {
        std::any v;
        Status status = ReadGGUFMetadataValue(gguf_file, elem_type, v);
        if (!status.OK()) {
          return status;
        }
        array_values.push_back(v);
      }
      any_value = array_values;
      break;
    }
    default:
      return Status(RET_INVALID_ARGUMENT, "Unsupported type encountered");
  }
  return Status();
}

std::string ReadStringFromFile(std::ifstream& gguf_file) {
  uint64_t length;
  gguf_file.read(reinterpret_cast<char*>(&length), sizeof(uint64_t));

  std::string value(length, '\0');
  gguf_file.read(&value[0], length);
  return value;
}

template <typename T>
Status ReadDataFromFile(std::ifstream& gguf_file, T& value) {
  T value_raw;
  if (!gguf_file.read(reinterpret_cast<char*>(&value_raw), sizeof(T))) {
    return Status(RET_INVALID_ARGUMENT, "Failed to read data from file.");
  }

  if constexpr (sizeof(T) == 1) {
    value = value_raw;
  } else if constexpr (sizeof(T) == 2) {
    value = le16toh(value_raw);
  } else if constexpr (sizeof(T) == 4) {
    value = le32toh(value_raw);
  } else if constexpr (sizeof(T) == 8) {
    value = le64toh(value_raw);
  }
  return Status();
}

DataType GetTensorDataType(uint32_t ggml_type) {
  switch (ggml_type) {
    case NewGGMLType::NEW_GGML_TYPE_F32:
      return TYPE_FP32;
    case NewGGMLType::NEW_GGML_TYPE_F16:
      return TYPE_FP16;
    case NewGGMLType::NEW_GGML_TYPE_I8:
      return TYPE_INT8;
    case NewGGMLType::NEW_GGML_TYPE_I16:
      return TYPE_INT16;
    case NewGGMLType::NEW_GGML_TYPE_I32:
      return TYPE_INT32;
    case NewGGMLType::NEW_GGML_TYPE_I64:
      return TYPE_INT64;
    case NewGGMLType::NEW_GGML_TYPE_F64:
      return TYPE_FP64;
    case NewGGMLType::NEW_GGML_TYPE_BF16:
      return TYPE_BF16;
    default:
      return TYPE_INVALID;
  }
}

Status ReadMeta(std::ifstream& gguf_file, std::shared_ptr<NewGGUFContext> gguf_context,
                std::unordered_map<std::string, NewGGUFMetaValue>& result) {
  gguf_file.read(reinterpret_cast<char*>(&gguf_context->header.magic), sizeof(uint32_t));
  if (gguf_context->header.magic != GGUF_MAGIC) {
    return Status(RET_INVALID_ARGUMENT, "Invalid GGUF magic number.");
  }

  gguf_file.read(reinterpret_cast<char*>(&gguf_context->header.version), sizeof(uint32_t));
  if (gguf_context->header.version != GGUF_VERSION) {
    return Status(RET_INVALID_ARGUMENT, "Unsupported GGUF version.");
  }

  gguf_file.read(reinterpret_cast<char*>(&gguf_context->header.tensor_count), sizeof(uint64_t));
  gguf_file.read(reinterpret_cast<char*>(&gguf_context->header.metadata_kv_count), sizeof(uint64_t));

  for (uint64_t i = 0; i < gguf_context->header.metadata_kv_count; ++i) {
    NewGGUFMetaValue meta_data;
    uint64_t key_length;
    gguf_file.read(reinterpret_cast<char*>(&key_length), sizeof(uint64_t));
    if (key_length > MAX_STRING_LENGTH) {
      return Status(RET_INVALID_ARGUMENT, "Invalid key length in metadata.");
    }

    std::string key(key_length, '\0');
    gguf_file.read(reinterpret_cast<char*>(&key[0]), key_length);

    uint32_t value_type_int;
    gguf_file.read(reinterpret_cast<char*>(&value_type_int), sizeof(uint32_t));
    meta_data.type = static_cast<NewGGUFMetaValueType>(value_type_int);
    Status status = ReadGGUFMetadataValue(gguf_file, meta_data.type, meta_data.value);
    if (!status.OK()) {
      return status;
    }
    result[key] = meta_data;
  }

  return Status();
}

Status GGUFFileLoader::GetMetaDict(std::unordered_map<std::string, NewGGUFMetaValue>& result) {
  std::shared_ptr<NewGGUFContext> gguf_context = std::make_shared<NewGGUFContext>();

  std::ifstream gguf_file(filename_, std::ios::binary | std::ios::ate);
  if (!gguf_file.is_open()) {
    return Status(RET_INVALID_ARGUMENT, FormatStr("Failed to open GGUF file %s.", filename_.c_str()));
  }

  gguf_file.seekg(0, std::ios::beg);

  Status status = ReadMeta(gguf_file, gguf_context, result);
  gguf_file.close();

  return status;
}

Status GGUFFileLoader::LoadGGUFModelMeta() {
  if (!loaded_) {
    gguf_context_ = std::make_shared<NewGGUFContext>();

    gguf_file_.open(filename_, std::ios::binary | std::ios::ate);
    if (!gguf_file_.is_open()) {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Failed to open GGUF file %s.", filename_.c_str()));
    }

    file_size_ = gguf_file_.tellg();
    gguf_file_.seekg(0, std::ios::beg);

    Status status = ReadMeta(gguf_file_, gguf_context_, gguf_context_->metadata_map);
    if (!status.OK()) {
      gguf_file_.close();
      return status;
    }

    loaded_ = true;
  }

  return Status();
}

Status GGUFFileLoader::LoadWeightNames(std::vector<std::string>& weight_names) {
  Status status = LoadGGUFModelMeta();
  if (!status.OK()) {
    return status;
  }

  for (uint64_t i = 0; i < gguf_context_->header.tensor_count; ++i) {
    NewGGUFTensorInfo tensor_info;
    tensor_info.name = ReadStringFromFile(gguf_file_);
    weight_names.push_back(tensor_info.name);

    status = ReadDataFromFile<uint32_t>(gguf_file_, tensor_info.n_dims);
    if (!status.OK()) {
      gguf_file_.close();
      return status;
    }

    for (uint32_t j = 0; j < tensor_info.n_dims; ++j) {
      uint64_t v;
      status = ReadDataFromFile<uint64_t>(gguf_file_, v);
      if (!status.OK()) {
        gguf_file_.close();
        return status;
      }
      tensor_info.dims.push_back(v);
    }

    uint32_t data_type_int;
    status = ReadDataFromFile<uint32_t>(gguf_file_, data_type_int);
    if (!status.OK()) {
      gguf_file_.close();
      return status;
    }

    tensor_info.data_type = GetTensorDataType(data_type_int);
    status = ReadDataFromFile<uint64_t>(gguf_file_, tensor_info.offset);
    if (!status.OK()) {
      gguf_file_.close();
      return status;
    }

    size_t tensor_data_size = 1;
    for (uint32_t dim : tensor_info.dims) {
      if (dim == 0 || tensor_data_size > SIZE_MAX / dim) {
        return Status(RET_INVALID_ARGUMENT, "Tensor size calculation overflow.");
      }
      tensor_data_size *= dim;
    }
    tensor_info.size = tensor_data_size * GetTypeSize(tensor_info.data_type);
    gguf_context_->tensor_info_map[tensor_info.name] = tensor_info;
  }

  return Status();
}

Status GGUFFileLoader::LoadModelWeights(const std::vector<std::string>& weight_names,
                                        std::unordered_map<std::string, Tensor>& result) {
  Status status = LoadGGUFModelMeta();
  if (!status.OK()) {
    return status;
  }

  std::unordered_set<std::string> weight_name_set(weight_names.begin(), weight_names.end());
  size_t alignment = gguf_context_->metadata_map.count("general.alignment")
                         ? std::any_cast<uint32_t>(gguf_context_->metadata_map["general.alignment"].value)
                         : GGUF_ALIGNMENT;

  auto offset = gguf_file_.tellg();
  size_t offset_pad = offset % alignment;
  if (offset_pad != 0) {
    offset += alignment - offset_pad;
  }
  gguf_context_->offset = offset;
  gguf_context_->alignment = alignment;

  if (gguf_context_->header.tensor_count > 0) {
    gguf_file_.seekg(gguf_context_->offset, std::ios::beg);

    size_t data_size = file_size_ - gguf_context_->offset;
    char* weights_buffer = new char[data_size];
    gguf_file_.read(weights_buffer, data_size);
    for (const auto& item : gguf_context_->tensor_info_map) {
      const NewGGUFTensorInfo& tensor_info = item.second;

      if (weight_name_set.find(tensor_info.name) == weight_name_set.end()) {
        KLLM_LOG_DEBUG << "Skip weight tensor name " << tensor_info.name;
        continue;
      }
      KLLM_LOG_DEBUG << "Load weight tensor name " << tensor_info.name;

      Tensor weight_tensor = Tensor(MemoryLocation::LOCATION_HOST, tensor_info.data_type, tensor_info.dims);

      std::string weight_name = tensor_info.name;
      memcpy(weight_tensor.GetPtr<void>(), weights_buffer + tensor_info.offset, weight_tensor.GetTotalBytes());
      result[weight_name] = weight_tensor;
    }
    delete[] weights_buffer;
  }
  gguf_file_.close();
  loaded_ = false;

  return Status();
}

}  // namespace ksana_llm
