/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"

namespace ksana_llm {

template <typename T>
Status EmbLookupLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "EmbLookupLayer not supported.");
}

template <typename T>
Status EmbLookupLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "EmbLookupLayer not supported.");
}
template class EmbLookupLayer<float>;
template class EmbLookupLayer<float16>;
template class EmbLookupLayer<bfloat16>;
}  // namespace ksana_llm
