/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/permute_layer.h"

namespace ksana_llm {

template <typename T>
Status PermuteLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "PermuteLayer not supported.");
}
template class PermuteLayer<float>;
template class PermuteLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class PermuteLayer<bfloat16>;
#endif
}  // namespace ksana_llm
