/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/concat_layer.h"

namespace ksana_llm {

template <typename T>
Status ConcatLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  concat_dim = std::any_cast<const size_t>(parameters[0]);
  return Status();
}

template <typename T>
Status ConcatLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("ConcatLayer not implement in Ascend.");
  return Status(RET_INFER_FAILED);
}
template class ConcatLayer<float>;
template class ConcatLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class ConcatLayer<bfloat16>;
#endif
}  // namespace ksana_llm
