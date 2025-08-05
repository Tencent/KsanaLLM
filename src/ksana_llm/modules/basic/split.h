/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class Split {
 public:
  explicit Split(const LayerCreationContext& creation_context);

  ~Split();

  Status Forward(Tensor input, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> split_layer_;
};
}  // namespace ksana_llm
