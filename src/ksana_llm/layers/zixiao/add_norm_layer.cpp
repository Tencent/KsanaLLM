/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_norm_layer.h"

namespace ksana_llm {

template <typename T>
Status AddNormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AddNormLayer not supported.");
}

template <typename T>
Status AddNormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "AddNormLayer not supported.");
}
template class AddNormLayer<float>;
template class AddNormLayer<float16>;
template class AddNormLayer<bfloat16>;

}  // namespace ksana_llm
