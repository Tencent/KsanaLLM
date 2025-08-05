/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/activation.h"

#include "ksana_llm/layers/activation_layer.h"

namespace ksana_llm {

Activation::Activation(const std::string& activation_type_str, const LayerCreationContext& creation_context) {
  ActivationType activation_type;
  if (activation_type_str == "gelu") {
    activation_type = ActivationType::Gelu;
  } else if (activation_type_str == "relu") {
    activation_type = ActivationType::Relu;
  } else if (activation_type_str == "geglu") {
    activation_type = ActivationType::Geglu;
  } else if (activation_type_str == "swiglu") {
    activation_type = ActivationType::Swiglu;
  } else {
    KLLM_THROW(fmt::format("Unsupport activation function: {}", activation_type_str));
  }
  activation_layer_ = std::make_shared<ActivationLayer>();
  activation_layer_->Init({activation_type}, creation_context.runtime_config, creation_context.context,
                          creation_context.rank);
}

Activation::~Activation() {}

Status Activation::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(activation_layer_->Forward(input_tensors, output_tensors));
  return Status();
}

}  // namespace ksana_llm
