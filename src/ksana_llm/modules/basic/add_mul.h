/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class AddMul {
 public:
  // Constructor for dual scale operations (MUL_THEN_ADD: input1 * scale1 + input2 * scale2)
  explicit AddMul(float scale1, float scale2, const LayerCreationContext& creation_context);

  ~AddMul();

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> add_mul_layer_;
};

}  // namespace ksana_llm