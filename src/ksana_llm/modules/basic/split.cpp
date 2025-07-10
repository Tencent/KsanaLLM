/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/split.h"

#include "ksana_llm/layers/split_layer.h"

namespace ksana_llm {

template <typename T>
Split<T>::Split(const LayerCreationContext<T>& creation_context) {
  split_layer_ = std::make_shared<SplitLayer<T>>();
  split_layer_->Init({}, creation_context.context, creation_context.rank);
}

template <typename T>
Split<T>::~Split() {}

template <typename T>
Status Split<T>::Forward(Tensor input, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(split_layer_->Forward({input}, output_tensors));
  return Status();
}

template class Split<float>;
template class Split<float16>;
#ifdef ENABLE_BFLOAT16
template class Split<bfloat16>;
#endif

}  // namespace ksana_llm
