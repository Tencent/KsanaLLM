/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/mul.h"

#include "ksana_llm/layers/mul_layer.h"

namespace ksana_llm {

Mul::Mul(const LayerCreationContext& creation_context) {
  mul_layer_ = std::make_shared<MulLayer>();
  mul_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status Mul::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(mul_layer_->Forward({A, B}, output_tensors));
  return Status();
}

}  // namespace ksana_llm
