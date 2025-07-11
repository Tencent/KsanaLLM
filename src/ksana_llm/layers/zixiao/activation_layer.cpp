/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/activation_layer.h"

namespace ksana_llm {

template <ActivationType ACTIVATION_TYPE, typename T>
Status ActivationLayer<ACTIVATION_TYPE, T>::Forward(const std::vector<Tensor>& input_tensors,
                                                    std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "ActivationLayer not supported.");
}

template class ActivationLayer<ActivationType::Gelu, float>;
template class ActivationLayer<ActivationType::Gelu, float16>;
template class ActivationLayer<ActivationType::Gelu, bfloat16>;

template class ActivationLayer<ActivationType::Relu, float>;
template class ActivationLayer<ActivationType::Relu, float16>;
template class ActivationLayer<ActivationType::Relu, bfloat16>;

template class ActivationLayer<ActivationType::Geglu, float>;
template class ActivationLayer<ActivationType::Geglu, float16>;
template class ActivationLayer<ActivationType::Geglu, bfloat16>;

template class ActivationLayer<ActivationType::Swiglu, float>;
template class ActivationLayer<ActivationType::Swiglu, float16>;
template class ActivationLayer<ActivationType::Swiglu, bfloat16>;

}  // namespace ksana_llm
