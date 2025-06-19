/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class Add {
 public:
  explicit Add(const LayerCreationContext<T>& creation_context, const std::string& weight_name = "");

  ~Add();

  Status Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors);

  Status Forward(Tensor A, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> add_layer_;
  Tensor weight_;
  bool with_weight_ = false;
};
}  // namespace ksana_llm
