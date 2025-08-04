/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/sigmoid.h"

#include "ksana_llm/layers/activation_layer.h"

namespace ksana_llm {

template <typename T>
Sigmoid<T>::Sigmoid(const LayerCreationContext& creation_context) {
  sigmoid_layer_ = std::make_shared<SigmoidLayer<T>>();
  sigmoid_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

template <typename T>
Status Sigmoid<T>::Forward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(sigmoid_layer_->Forward(input_tensors, output_tensors));
  return Status();
}

template class Sigmoid<float>;
template class Sigmoid<float16>;
template class Sigmoid<bfloat16>;

}  // namespace ksana_llm
