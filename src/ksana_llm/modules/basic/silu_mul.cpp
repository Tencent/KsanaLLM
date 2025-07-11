/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/silu_mul.h"

#include "ksana_llm/layers/silu_mul_layer.h"

namespace ksana_llm {

template <typename T>
SiluMul<T>::SiluMul(const LayerCreationContext<T>& creation_context) {
  silu_mul_layer_ = std::make_shared<SiluMulLayer<T>>();
  silu_mul_layer_->Init({}, creation_context.context, creation_context.rank);
}

template <typename T>
SiluMul<T>::~SiluMul() {}

template <typename T>
Status SiluMul<T>::Forward(Tensor bias, Tensor gated_bias, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({bias, gated_bias}, output_tensors));
  return Status();
}

template <typename T>
Status SiluMul<T>::Forward(Tensor fused_tensor, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(silu_mul_layer_->Forward({fused_tensor}, output_tensors));
  return Status();
}

template class SiluMul<float>;
template class SiluMul<float16>;
template class SiluMul<bfloat16>;

}  // namespace ksana_llm
