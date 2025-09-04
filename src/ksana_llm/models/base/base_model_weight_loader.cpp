/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include <c10/core/ScalarType.h>
#include <dlfcn.h>
#include <torch/types.h>
#include <algorithm>

#include "fmt/core.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

BaseModelWeightLoader::BaseModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                             std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  model_config_ = model_config;
  env_ = env;
  context_ = context;

  permute_buffers_.resize(context_->GetTensorParallelSize());
  tensor_buffers_.resize(context_->GetTensorParallelSize());
}

BaseModelWeightLoader::~BaseModelWeightLoader() {}

Status BaseModelWeightLoader::FilterModelFiles(std::vector<std::string>& model_files) { return Status(); }

Status BaseModelWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) { return Status(); }

Status BaseModelWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
  return Status();
}

Status BaseModelWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                      int dev_rank) {
  return Status();
}

Status BaseModelWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                  int dev_rank,
                                                  std::unordered_map<std::string, Tensor>& device_model_weights,
                                                  std::unordered_map<std::string, Tensor>& left_host_weights) {
  return Status();
}

bool BaseModelWeightLoader::CheckWeightNameMatched(const std::string& weight_name,
                                                   const std::vector<std::string>& name_list, bool full_match) {
  if (full_match) {
    return std::find(name_list.begin(), name_list.end(), weight_name) != name_list.end();
  }

  for (const auto& name_piece : name_list) {
    if (weight_name.find(name_piece) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool BaseModelWeightLoader::CheckAllWeightsExist(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                 const std::vector<std::string>& name_list) {
  for (const auto& name : name_list) {
    if (host_model_weights.find(name) == host_model_weights.end()) {
      return false;
    }
  }
  return true;
}

Status BaseModelWeightLoader::CopyHostTensorToDevice(const Tensor host_tensor, int dev_rank, Tensor& dev_tensor) {
  dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_tensor.dtype, host_tensor.shape, dev_rank);

  MemcpyAsync(dev_tensor.GetPtr<void>(), host_tensor.GetPtr<void>(), host_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[dev_rank]);

  return Status();
}

Status BaseModelWeightLoader::PermuteDeviceTensor(const Tensor& input_tensor, const std::vector<size_t>& permutation,
                                                  int dev_rank, Tensor& output_tensor) {
  Permute(const_cast<Tensor&>(input_tensor), output_tensor, permutation, context_->GetMemoryManageStreams()[dev_rank]);
  output_tensor.shape = input_tensor.shape;
  for (size_t i = 0; i < permutation.size(); ++i) {
    output_tensor.shape[i] = input_tensor.shape[permutation[i]];
  }

  // For compatible with Ascend device.
  TransLayout(output_tensor, context_->GetMemoryManageStreams()[dev_rank]);
  return Status();
}

Status BaseModelWeightLoader::GetCustomWeightMap(const std::string& model_path, const std::string& model_type,
                                                 std::unordered_map<std::string, Tensor>& weight_map,
                                                 ModelFormat model_format) {
  if (patterns_.empty()) {
    return Status();
  }

  std::vector<std::string> weight_names;
  weight_names.reserve(weight_map.size());
  for (const auto& [key, value] : weight_map) {
    weight_names.push_back(key);
  }

  for (const std::string& weight_name : weight_names) {
    for (const auto& [pattern, format] : patterns_) {
      if (std::regex_search(weight_name, pattern)) {
        std::string custom_name = std::regex_replace(weight_name, pattern, format);
        auto node = weight_map.extract(weight_name);
        node.key() = std::move(custom_name);
        weight_map.insert(std::move(node));
        break;
      }
    }
  }

  return Status();
}

std::string BaseModelWeightLoader::GetWeightMapPath(const std::string& model_path, const std::string& model_type,
                                                    ModelFormat model_format) {
  OptionalFile optional_file;
  std::string weight_map_suffix = "_weight_map.json";
  if (model_format == ModelFormat::GGUF) {
    weight_map_suffix = "_gguf_weight_map.json";
  }
  std::string& weight_path = optional_file.GetOptionalFile(model_path, "weight_map", model_type + weight_map_suffix);
  if (weight_path.empty()) {  // searching weight_map under ksana_llm
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&BaseModelWeightLoader::GetWeightMapPath), &info)) {
      std::filesystem::path so_path = std::filesystem::absolute(info.dli_fname);
      const std::string kRelativePath = "../../src/ksana_llm/python";
      std::string ksana_weight_map_path = so_path.parent_path() / kRelativePath / "weight_map";
      weight_path = optional_file.GetOptionalFile(ksana_weight_map_path, "weight_map", model_type + weight_map_suffix);
    }
  }
  return weight_path;
}

Status BaseModelWeightLoader::InitRegexPatterns(const std::string& model_path, const std::string& model_type,
                                                ModelFormat model_format) {
  if (loaded_patterns_) {
    return Status();
  }
  const std::string weight_path = GetWeightMapPath(model_path, model_type, model_format);
  if (!weight_path.empty()) {
    std::ifstream file(weight_path);
    if (!file.is_open()) {
      KLLM_LOG_ERROR << fmt::format("Load weight map json: {} error.", weight_path) << std::endl;
      return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load weight map json: {} error.", weight_path));
    }
    const nlohmann::json& weight_map_json = nlohmann::json::parse(file);
    file.close();

    patterns_.clear();
    patterns_.reserve(weight_map_json.size());
    for (const auto& [key, format] : weight_map_json.items()) {
      patterns_.emplace_back(std::regex(key), format.get<std::string>());
    }
    loaded_patterns_ = true;
  }
  return Status();
}

// TODO(huicongyao): support tensor type cast for Ascend
Status BaseModelWeightLoader::CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank) {
// Temporarily not support cast for Ascend and tops device
#ifdef ENABLE_ACL
  KLLM_THROW(fmt::format("Unsupported tensor cast for Ascend device"));
#elif ENABLE_TOPS
  KLLM_THROW(fmt::format("Unsupported tensor cast for Tops device"));
#endif
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }
#ifdef ENABLE_CUDA
  if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_FP16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToHalf(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_BF16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToBFloat16(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<bfloat16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(input_tensor.GetPtr<float16>(),
                                                           input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
                                                           context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(
        input_tensor.GetPtr<bfloat16>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToHalf(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_BF16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToBFloat16(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {}", input_tensor.dtype, new_dtype));
  }
#endif
  return Status();
}

Status BaseModelWeightLoader::CastDeviceTensorTypePytorch(Tensor& input_tensor, DataType new_dtype, int dev_rank) {
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }

  torch::ScalarType torch_dtype;
  if (input_tensor.dtype == DataType::TYPE_FP32) {
    torch_dtype = torch::kFloat32;
  } else if (input_tensor.dtype == DataType::TYPE_FP16) {
    torch_dtype = torch::kFloat16;
  } else if (input_tensor.dtype == DataType::TYPE_BF16) {
    torch_dtype = torch::kBFloat16;
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor type {}", input_tensor.dtype));
  }

  auto options = torch::TensorOptions().device(torch::kCUDA, dev_rank).dtype(torch_dtype);
  torch::Tensor torch_input_tensor =
      torch::from_blob(input_tensor.GetPtr<void>(), {static_cast<int64_t>(input_tensor.GetElementNumber())}, options);

  if (GetTypeSize(new_dtype) > GetTypeSize(input_tensor.dtype)) {
    input_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
  }
  input_tensor.dtype = new_dtype;

  // Sync before torch operation.
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  torch::Tensor torch_output_tensor;
  if (new_dtype == TYPE_FP32) {
    torch_output_tensor = torch_input_tensor.to(torch::kFloat32);
  } else if (new_dtype == TYPE_FP16) {
    torch_output_tensor = torch_input_tensor.to(torch::kFloat16);
  } else if (new_dtype == TYPE_BF16) {
    torch_output_tensor = torch_input_tensor.to(torch::kBFloat16);
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor type {}", new_dtype));
  }

  MemcpyAsync(input_tensor.GetPtr<void>(), torch_output_tensor.data_ptr(), input_tensor.GetTotalBytes(),
              MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  return Status();
}

}  // namespace ksana_llm
