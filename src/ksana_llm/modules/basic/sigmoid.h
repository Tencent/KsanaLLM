/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class Sigmoid {
 public:
  // Disable a default constructor
  explicit Sigmoid(const LayerCreationContext<T>& creation_context);

  ~Sigmoid() = default;

  Status Forward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> sigmoid_layer_;
};
}  // namespace ksana_llm
