/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/activation.h"

#include "ksana_llm/layers/activation_layer.h"

namespace ksana_llm {

template <typename T>
Activation<T>::Activation(const std::string& activation_type, const LayerCreationContext& creation_context) {
  if (activation_type == "gelu") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Gelu, T>>();
  } else if (activation_type == "relu") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Relu, T>>();
  } else if (activation_type == "geglu") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Geglu, T>>();
  } else if (activation_type == "swiglu") {
    activation_layer_ = std::make_shared<ActivationLayer<ActivationType::Swiglu, T>>();
  } else {
    KLLM_THROW(fmt::format("Unsupport activation function: {}", activation_type));
  }
  activation_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

template <typename T>
Activation<T>::~Activation() {}

template <typename T>
Status Activation<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(activation_layer_->Forward(input_tensors, output_tensors));
  return Status();
}

template class Activation<float>;
template class Activation<float16>;
template class Activation<bfloat16>;

}  // namespace ksana_llm
