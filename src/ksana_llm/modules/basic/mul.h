/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class Mul {
 public:
  // Disable a default constructor
  explicit Mul(const LayerCreationContext<T>& creation_context);

  ~Mul() = default;

  Status Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> mul_layer_;
};
}  // namespace ksana_llm
