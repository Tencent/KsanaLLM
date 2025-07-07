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
  Status AddWeightTensor(const std::string& weight_name, const std::vector<size_t>& shapes, const DataType& dtype) {
    const auto& [it, is_new] =
        weights_map_.try_emplace(weight_name, Tensor(MemoryLocation::LOCATION_DEVICE, dtype, shapes, rank_));
    if (is_new) {
      KLLM_LOG_DEBUG << "TensorManager::AddWeightTensor, create weight " << weight_name;
    } else {
      KLLM_LOG_WARNING << fmt::format("The weight named {} has already been created. Skip creating the weight tensor.",
                                      weight_name);
    }
    return Status();
  }

  // Create a tensor with the same size, similar to ```copy_tensor_name = torch.empty_like(origin_tensor_name)```
  Status CreateTensorWithSameShape(const std::string& origin_tensor_name, const std::string& copy_tensor_name) {
    const auto origin_it = weights_map_.find(origin_tensor_name);
    if (origin_it == weights_map_.end()) {
      KLLM_THROW(
          fmt::format("Create tensor {} faild: tensor {} not in weights map", copy_tensor_name, origin_tensor_name));
    }
    const Tensor& origin_tensor = origin_it->second;
    AddWeightTensor(copy_tensor_name, origin_tensor.shape, origin_tensor.dtype);
    return Status();
  }

 private:
  int rank_ = 0;
  std::unordered_map<std::string, Tensor>& weights_map_;
};

}  // namespace ksana_llm
