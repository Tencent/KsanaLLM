/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/split_layer.h"

namespace ksana_llm {

template <typename T>
Status SplitLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}

template <typename T>
Status SplitLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}
template class SplitLayer<float>;
template class SplitLayer<float16>;
template class SplitLayer<bfloat16>;
}  // namespace ksana_llm
