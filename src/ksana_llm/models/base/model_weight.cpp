/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_weight.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

ModelWeight::ModelWeight() {}

ModelWeight::~ModelWeight() {}

const Tensor& ModelWeight::GetWeightTensor(const std::string& weight_name) const {
  auto it = weights_map_.find(weight_name);
  if (it == weights_map_.end()) {
    KLLM_LOG_WARNING << fmt::format("weight_name: {} not in weights map", weight_name);
    return empty_tensor_;
  }
  return it->second;
}

std::vector<std::string> ModelWeight::GetWeightNames() const {
  std::vector<std::string> weight_names;
  weight_names.reserve(weights_map_.size());
  for (auto it = weights_map_.begin(); it != weights_map_.end(); ++it) {
    weight_names.push_back(it->first);
  }
  return weight_names;
}

}  // namespace ksana_llm
