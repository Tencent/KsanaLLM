/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_weight.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

ModelWeight::ModelWeight() {}

ModelWeight::~ModelWeight() {}

const Tensor & ModelWeight::GetWeightTensor(const std::string& weight_name) const {
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

Status ModelWeight::BindQuantWeightScales() {
  for (auto & [weight_name, weight_tensor] : weights_map_) {
    if (weight_name.find("proj.weight") != std::string::npos &&
        weight_name.find("proj.weight_scale_inv") == std::string::npos) {
      std::string weight_scale_name = weight_name + "_scale_inv";
      auto scale_iter = weights_map_.find(weight_scale_name);
      if (scale_iter != weights_map_.end()) {
        weight_tensor.weight_scales = &(scale_iter->second);
        KLLM_LOG_INFO << fmt::format("bind {}, shape: {} to {}, shape: {}\n",
                                      weight_scale_name,
                                      Vector2Str(std::vector<size_t>(weight_tensor.weight_scales->shape)),
                                      weight_name,
                                      Vector2Str(std::vector<size_t>(weight_tensor.shape)));
      } else {
        KLLM_LOG_INFO << fmt::format("weight scale not found: {}", weight_scale_name);
      }
    }
  }
  return Status();
}

}  // namespace ksana_llm
