/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"

#include <cstdint>

namespace ksana_llm {

template <typename T>
Status LayernormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}

template <typename T>
Status LayernormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "LayernormLayer not supported.");
}
template class LayernormLayer<float>;
template class LayernormLayer<float16>;
template class LayernormLayer<bfloat16>;
}  // namespace ksana_llm
