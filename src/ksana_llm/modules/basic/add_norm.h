/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

// Fused Add and RMS Norm operation (currently Layernorm is not supported)
template <typename T>
class AddNorm {
 public:
  // Disable a default constructor
  AddNorm(const std::string& weight_name, float norm_eps, const LayerCreationContext<T>& creation_context);

  ~AddNorm();

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> add_norm_layer_;
  Tensor weight_;
};
}  // namespace ksana_llm
