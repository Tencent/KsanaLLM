/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include <c10/core/ScalarType.h>
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
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

BaseModelWeightLoader::BaseModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                             std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  model_config_ = model_config;
  env_ = env;
  context_ = context;
}

BaseModelWeightLoader::~BaseModelWeightLoader() {}

Status BaseModelWeightLoader::FilterModelFiles(std::vector<std::string>& model_files) { return Status(); }

Status BaseModelWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) { return Status(); }

Status BaseModelWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
  return Status();
}

Status BaseModelWeightLoader::PostProcessModelWeights(
    std::unordered_map<std::string, Tensor>& dev_weights_map, int dev_rank) {
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
  // output_tensor.shape = {input_tensor.shape[1], input_tensor.shape[0]};
  output_tensor.shape = input_tensor.shape;
  for (size_t i = 0; i < permutation.size(); ++i) {
    output_tensor.shape[i] = input_tensor.shape[permutation[i]];
  }

  // For compatible with Ascend device.
  TransLayout(output_tensor, context_->GetMemoryManageStreams()[dev_rank]);
  return Status();
}

Status BaseModelWeightLoader::CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank) {
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
