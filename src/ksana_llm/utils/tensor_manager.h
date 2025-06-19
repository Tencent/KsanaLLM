/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

// Tensor Manager, using in models/xxx/xxx_weight.cpp
// Providing the functionality to load weights into the weight list.
// Each device needs to maintain one.
class TensorManager {
 public:
  TensorManager(int rank, std::unordered_map<std::string, Tensor>& weights_map)
      : rank_(rank), weights_map_(weights_map) {}
  ~TensorManager() {}

  // Create a new weight in the weight list located on DEVICE
  // Note that it does not contain real data when created and needs to be copied manually.
  // TODO(jinxcwu): Add device option, do not force creation on GPU.
  Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype) {
    if (weights_map_.count(weight_name)) {
      KLLM_LOG_WARNING << fmt::format("The weight named {} has already been created. Skip creating the weight tensor.",
                                      weight_name);
      return Status();
    }

    KLLM_LOG_DEBUG << "TensorManager::AddWeightTensor, create weight " << weight_name;
    weights_map_.emplace(weight_name, Tensor(MemoryLocation::LOCATION_DEVICE, dtype, shapes, rank_));
    return Status();
  }

  // Create a tensor with the same size, similar to ```copy_tensor_name = torch.empty_like(origin_tensor_name)```
  Status CreateTensorWithSameShape(const std::string& origin_tensor_name, const std::string& copy_tensor_name) {
    if (!weights_map_.count(origin_tensor_name)) {
      KLLM_THROW(
          fmt::format("Create tensor {} faild: tensor {} not in weights map", copy_tensor_name, origin_tensor_name));
    }
    Tensor& origin_tensor = weights_map_[origin_tensor_name];
    AddWeightTensor(copy_tensor_name, origin_tensor.shape, origin_tensor.dtype);
    return Status();
  }

 private:
  int rank_ = 0;
  std::unordered_map<std::string, Tensor>& weights_map_;
};

}  // namespace ksana_llm
