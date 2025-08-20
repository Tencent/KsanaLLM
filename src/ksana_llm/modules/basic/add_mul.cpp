/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/add_mul.h"

#include "ksana_llm/layers/add_mul_layer.h"

namespace ksana_llm {

AddMul::AddMul(float scale1, float scale2, const LayerCreationContext& creation_context) {
  add_mul_layer_ = std::make_shared<AddMulLayer>();
  add_mul_layer_->Init({scale1, scale2}, creation_context.runtime_config, creation_context.context,
                       creation_context.rank);
}

AddMul::~AddMul() {}

Status AddMul::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_mul_layer_->Forward(input_tensors, output_tensors));
  return Status();
}

}  // namespace ksana_llm