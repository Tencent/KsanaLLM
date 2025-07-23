/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

namespace ksana_llm {

template <typename T>
Status AddLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AddLayer not supported.");
}

template <typename T>
Status AddLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "AddLayer not supported.");
}
template class AddLayer<float>;
template class AddLayer<float16>;
template class AddLayer<bfloat16>;

}  // namespace ksana_llm
