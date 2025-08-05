/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/sigmoid.h"

#include "ksana_llm/layers/activation_layer.h"

namespace ksana_llm {

Sigmoid::Sigmoid(const LayerCreationContext& creation_context) {
  sigmoid_layer_ = std::make_shared<SigmoidLayer>();
  sigmoid_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Status Sigmoid::Forward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(sigmoid_layer_->Forward(input_tensors, output_tensors));
  return Status();
}

}  // namespace ksana_llm
