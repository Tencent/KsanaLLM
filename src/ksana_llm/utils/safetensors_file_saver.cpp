/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/safetensors_file_saver.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "ksana_llm/utils/device_types.h"

using json = nlohmann::json;

namespace ksana_llm {

SafetensorsFileSaver::SafetensorsFileSaver(const std::string& base_file_name, size_t rank,
                                           std::shared_ptr<Context> context, size_t max_file_size_bytes)
    : base_file_name_(base_file_name), max_file_size_bytes_(max_file_size_bytes), rank_(rank), context_(context) {
  SetDevice(rank_);
}

std::string SafetensorsFileSaver::ConvertDataTypeToSafetensorsDtype(DataType data_type) {
  switch (data_type) {
    case TYPE_FP16:
      return "F16";
    case TYPE_FP32:
      return "F32";
    case TYPE_BF16:
      return "BF16";
    case TYPE_INT32:
      return "I32";
    case TYPE_FP8_E4M3:
      return "F8_E4M3";
    case TYPE_UINT8:
      return "UI8";
    default:
      KLLM_THROW(fmt::format("Unsupported data type: {}", GetTypeString(data_type)));
  }
}

std::vector<std::vector<std::string>> SafetensorsFileSaver::GroupTensorsIntoFiles(
    const std::unordered_map<std::string, Tensor>& weights_map) {
  // Calculate size of each tensor
  std::unordered_map<std::string, size_t> tensor_sizes;
  size_t total_size = 0;
  for (const auto& [name, tensor] : weights_map) {
    size_t tensor_size = tensor.GetTotalBytes();
    tensor_sizes[name] = tensor_size;
    total_size += tensor_size;
  }

  // Group tensors into files
  std::vector<std::vector<std::string>> file_groups;
  std::vector<std::string> current_group;
  size_t current_group_size = 0;

  // Sort tensors by size (largest first) to optimize packing
  std::vector<std::pair<std::string, size_t>> sorted_tensors;
  for (const auto& [name, size] : tensor_sizes) {
    sorted_tensors.push_back({name, size});
  }
  std::sort(sorted_tensors.begin(), sorted_tensors.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  // First handle any tensors that are individually larger than max_file_size_bytes_
  for (auto it = sorted_tensors.begin(); it != sorted_tensors.end();) {
    if (it->second > max_file_size_bytes_) {
      std::vector<std::string> large_tensor_group = {it->first};
      file_groups.push_back(large_tensor_group);
      it = sorted_tensors.erase(it);
    } else {
      ++it;
    }
  }
  // Group remaining tensors
  for (const auto& [name, size] : sorted_tensors) {
    if (current_group_size + size > max_file_size_bytes_ && !current_group.empty()) {
      file_groups.push_back(current_group);
      current_group.clear();
      current_group_size = 0;
    }
    current_group.push_back(name);
    current_group_size += size;
  }
  // Add the last group if not empty
  if (!current_group.empty()) {
    file_groups.push_back(current_group);
  }
  KLLM_LOG_INFO << fmt::format("Grouped tensors into {} files", file_groups.size());
  return file_groups;
}

Status SafetensorsFileSaver::SaveTensorGroup(const std::vector<std::string>& tensor_names,
                                             const std::unordered_map<std::string, Tensor>& weights_map,
                                             const std::string& file_name) {
  // Create JSON metadata
  json metadata;

  // Calculate total data size and offsets
  size_t total_data_size = 0;
  std::unordered_map<std::string, std::pair<size_t, size_t>> offsets;  // {tensor_name: {start_offset, end_offset}}
  for (const auto& tensor_name : tensor_names) {
    if (weights_map.find(tensor_name) == weights_map.end()) {
      KLLM_LOG_ERROR << fmt::format("Tensor {} not found in weights_map", tensor_name);
      continue;
    }
    const auto& tensor = weights_map.at(tensor_name);
    size_t tensor_size = tensor.GetTotalBytes();
    offsets[tensor_name] = {total_data_size, total_data_size + tensor_size};
    total_data_size += tensor_size;
  }

  // Build metadata JSON
  for (const auto& tensor_name : tensor_names) {
    if (offsets.find(tensor_name) == offsets.end()) {
      continue;
    }
    const auto& tensor = weights_map.at(tensor_name);
    DataType data_type = tensor.dtype;

    // Convert shape to JSON array
    json shape_json = json::array();
    for (size_t dim : std::vector<size_t>(tensor.shape)) {
      shape_json.push_back(dim);
    }

    // Add tensor metadata to JSON
    metadata[tensor_name] = {{"dtype", ConvertDataTypeToSafetensorsDtype(data_type)},
                             {"shape", shape_json},
                             {"data_offsets", {offsets[tensor_name].first, offsets[tensor_name].second}}};
  }

  // Convert metadata to string
  std::string metadata_str = metadata.dump();

  std::filesystem::path file_path(file_name);
  std::filesystem::path directory_path = file_path.parent_path();
  std::filesystem::create_directories(directory_path);
  std::ofstream file(file_name, std::ios::binary);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Failed to open file for writing: {}", file_name);
    return Status(RET_RUNTIME_FAILED, fmt::format("Failed to open file for writing: {}", file_name));
  }

  // Write header size
  uint64_t header_size = metadata_str.size();
  file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint64_t));

  // Write metadata
  file.write(metadata_str.c_str(), metadata_str.size());

  // Write tensor data
  for (const auto& tensor_name : tensor_names) {
    if (offsets.find(tensor_name) == offsets.end()) {
      continue;
    }
    const auto& tensor = weights_map.at(tensor_name);
    if (tensor.location == MemoryLocation::LOCATION_HOST) {
      void* tensor_data = tensor.GetPtr<void>();
      size_t tensor_size = tensor.GetTotalBytes();
      file.write(static_cast<const char*>(tensor_data), tensor_size);
    } else {  // DEVICE
      size_t tensor_size = tensor.GetTotalBytes();
      void* host_data = malloc(tensor_size);
      if (!host_data) {
        return Status(RET_RUNTIME_FAILED, fmt::format("Failed to allocate host memory for tensor {}", tensor_name));
      }
      void* device_data = tensor.GetPtr<void>();
      if (device_data) {
        Memcpy(host_data, device_data, tensor_size, MEMCPY_DEVICE_TO_HOST);
      } else {
        free(host_data);
        return Status(RET_RUNTIME_FAILED, fmt::format("Failed to copy tensor {}", tensor_name));
      }
      file.write(static_cast<const char*>(host_data), tensor_size);
      free(host_data);
    }
  }

  file.close();
  KLLM_LOG_INFO << fmt::format("Saved {} tensors to {}", tensor_names.size(), file_name);
  return Status();
}

Status SafetensorsFileSaver::SaveTensors(const std::unordered_map<std::string, Tensor>& weights_map) {
  // Group tensors into files
  auto file_groups = GroupTensorsIntoFiles(weights_map);
  // Save each group to a separate file
  for (size_t i = 0; i < file_groups.size(); ++i) {
    std::string file_name = base_file_name_ + "_" + std::to_string(i) + ".safetensors";
    Status status = SaveTensorGroup(file_groups[i], weights_map, file_name);
    if (!status.OK()) {
      KLLM_LOG_ERROR << fmt::format("Failed to save tensor group to {}: {}", file_name, status.ToString());
      return status;
    }
  }
  KLLM_LOG_INFO << fmt::format("Successfully saved all tensors to {} files", file_groups.size());
  return Status();
}

}  // namespace ksana_llm
