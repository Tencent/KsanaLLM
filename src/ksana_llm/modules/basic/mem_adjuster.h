/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/mem_adjuster_layer.h"
#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class MemAdjuster {
 public:
  explicit MemAdjuster(const LayerCreationContext& creation_context);

  ~MemAdjuster();

  // Extract a subset of columns from a 2D tensor
  // Input: tensor of shape (m, n)
  // Output: tensor[:, input_offset:input_offset + output_n] with shape (m, output_n)
  Status ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset, size_t output_n);

 protected:
  std::shared_ptr<MemAdjusterLayer> mem_adjuster_layer_;
};
}  // namespace ksana_llm
