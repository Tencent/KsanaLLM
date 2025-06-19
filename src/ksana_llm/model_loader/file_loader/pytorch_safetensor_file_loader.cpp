/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/file_loader/pytorch_safetensor_file_loader.h"

#include <filesystem>
#include <unordered_set>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

PytorchSafetensorFileLoader::PytorchSafetensorFileLoader(const std::string& filename) : filename_(filename) {}

PytorchSafetensorFileLoader::~PytorchSafetensorFileLoader() {
  if (safetensors_file_.is_open()) {
    safetensors_file_.close();
    loaded_ = false;
  }
}

DataType GetTensorDataType(const std::string& safetensor_dtype) {
  if (safetensor_dtype == "F16") {
    return TYPE_FP16;
  } else if (safetensor_dtype == "F32") {
    return TYPE_FP32;
  } else if (safetensor_dtype == "BF16") {
    return TYPE_BF16;
  } else if (safetensor_dtype == "I32") {
    return TYPE_INT32;
  } else if (safetensor_dtype == "F8_E4M3") {
    return TYPE_FP8_E4M3;
  }
  return TYPE_INVALID;
}

Status PytorchSafetensorFileLoader::LoadSafetensorTensorDict() {
  if (!loaded_) {
    safetensors_file_.open(filename_, std::ios::binary | std::ios::ate);
    if (!safetensors_file_.is_open()) {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Failed to load safetensors file %s.", filename_.c_str()));
    }

    int64_t file_size = safetensors_file_.tellg();
    if (file_size == -1) {
      safetensors_file_.close();
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("Invalid safetensors file size: -1, filename: %s", filename_.c_str()));
    }
    safetensors_file_.seekg(0, std::ios::beg);

    // get the tensor list(string)
    size_t header_size;
    safetensors_file_.read(reinterpret_cast<char*>(&header_size), sizeof(size_t));

    std::string tensor_dict_str;
    tensor_dict_str.resize(header_size);
    safetensors_file_.read(&tensor_dict_str[0], header_size);
    KLLM_LOG_DEBUG << FormatStr("Safetensors file %s Header = %s", filename_.c_str(), tensor_dict_str.c_str());

    // Parsing JSON to retrieve tensor information.
    tensor_dict_ = json::parse(tensor_dict_str);

    loaded_ = true;
  }
  return Status();
}

Status PytorchSafetensorFileLoader::LoadWeightNames(std::vector<std::string>& weight_names) {
  Status status = LoadSafetensorTensorDict();
  if (!status.OK()) {
    return status;
  }

  for (const auto& tensor_iter : tensor_dict_.items()) {
    const std::string& weight_name = tensor_iter.key();
    json tensor_data = tensor_iter.value();
    if (!tensor_data.contains("data_offsets")) {
      continue;
    }
    weight_names.push_back(weight_name);
  }

  return Status();
}

Status PytorchSafetensorFileLoader::LoadModelWeights(const std::vector<std::string>& weight_names,
                                                     std::unordered_map<std::string, Tensor>& result) {
  Status status = LoadSafetensorTensorDict();
  if (!status.OK()) {
    return status;
  }

  std::unordered_set<std::string> weight_name_set(weight_names.begin(), weight_names.end());
  size_t last_end_index = 0;
  for (const auto& tensor_iter : tensor_dict_.items()) {
    const std::string& weight_name = tensor_iter.key();
    if (weight_name_set.find(weight_name) == weight_name_set.end()) {
      KLLM_LOG_DEBUG << "Skip weight tensor name " << weight_name;
      continue;
    }
    KLLM_LOG_DEBUG << "Load weight tensor name " << weight_name;

    std::vector<size_t> tensor_shape;
    json tensor_data = tensor_iter.value();
    for (size_t dim : tensor_data["shape"]) {
      tensor_shape.emplace_back(dim);
    }

    KLLM_LOG_DEBUG << FormatStr("SafeTensors Loader: weight_name:%s, shape:%s", weight_name.c_str(),
                                Vector2Str(tensor_shape).c_str());

    size_t tensor_beg_index = tensor_data["data_offsets"][0];
    size_t tensor_end_index = tensor_data["data_offsets"][1];
    size_t tensor_size = tensor_end_index - tensor_beg_index;

    DataType tensor_dtype = GetTensorDataType(tensor_data["dtype"]);

    Tensor weight_tensor = Tensor(MemoryLocation::LOCATION_HOST, tensor_dtype, tensor_shape);
    safetensors_file_.seekg(tensor_beg_index - last_end_index, std::ios::cur);
    safetensors_file_.read(weight_tensor.GetPtr<char>(), tensor_size);
    result[weight_name] = weight_tensor;
    last_end_index = tensor_end_index;
  }

  // Could read only once.
  safetensors_file_.close();
  loaded_ = false;

  return Status();
}

}  // namespace ksana_llm
