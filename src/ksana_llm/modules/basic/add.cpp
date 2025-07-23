/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/add.h"

#include "ksana_llm/layers/add_layer.h"

namespace ksana_llm {

template <typename T>
Add<T>::Add(const LayerCreationContext<T>& creation_context, const std::string& weight_name) {
  add_layer_ = std::make_shared<AddLayer<T>>();
  add_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
  if (weight_name != "") {
    weight_ = creation_context.base_weight->GetModelWeights(weight_name);
    with_weight_ = true;
  }
}

template <typename T>
Add<T>::~Add() {}

template <typename T>
Status Add<T>::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_layer_->Forward({A, B}, output_tensors));
  return Status();
}

template <typename T>
Status Add<T>::Forward(Tensor A, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_layer_->Forward({A, weight_}, output_tensors));
  return Status();
}

template class Add<float>;
template class Add<float16>;
template class Add<bfloat16>;

}  // namespace ksana_llm
