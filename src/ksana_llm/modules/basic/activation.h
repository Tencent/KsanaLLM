/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class Activation {
 public:
  // Disable a default constructor
  explicit Activation(const std::string& activation_type, const LayerCreationContext& creation_context);

  ~Activation();

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> activation_layer_;
};
}  // namespace ksana_llm
