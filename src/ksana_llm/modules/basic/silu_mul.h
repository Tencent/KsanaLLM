/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class SiluMul {
 public:
  // Disable a default constructor
  explicit SiluMul(const LayerCreationContext& creation_context);

  ~SiluMul();

  //  output_tensors = Silu(bias) * gated_bias
  Status Forward(Tensor bias, Tensor gated_bias, std::vector<Tensor>& output_tensors);

  // n = fused_tensor.shape[1]
  // output_tensors = Silu(fused_tensor[:, :n/2]) * fused_tensor[:, n/2:]
  Status Forward(Tensor fused_tensor, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> silu_mul_layer_;
};
}  // namespace ksana_llm
