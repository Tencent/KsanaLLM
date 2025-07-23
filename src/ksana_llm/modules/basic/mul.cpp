/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/mul.h"

#include "ksana_llm/layers/mul_layer.h"

namespace ksana_llm {

template <typename T>
Mul<T>::Mul(const LayerCreationContext<T>& creation_context) {
  mul_layer_ = std::make_shared<MulLayer<T>>();
  mul_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

template <typename T>
Status Mul<T>::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(mul_layer_->Forward({A, B}, output_tensors));
  return Status();
}

template class Mul<float>;
template class Mul<float16>;
template class Mul<bfloat16>;

}  // namespace ksana_llm
