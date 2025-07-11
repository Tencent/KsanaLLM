/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/add_norm.h"

#include "ksana_llm/layers/add_norm_layer.h"

namespace ksana_llm {

template <typename T>
AddNorm<T>::AddNorm(const std::string& weight_name, float norm_eps, const LayerCreationContext<T>& creation_context) {
  add_norm_layer_ = std::make_shared<AddNormLayer<T>>();
  add_norm_layer_->Init({norm_eps}, creation_context.context, creation_context.rank);
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

template <typename T>
AddNorm<T>::~AddNorm() {}

template <typename T>
Status AddNorm<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_norm_layer_->Forward({input_tensors[0], input_tensors[1], weight_}, output_tensors));
  return Status();
}

template class AddNorm<float>;
template class AddNorm<float16>;
template class AddNorm<bfloat16>;

}  // namespace ksana_llm
