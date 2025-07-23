/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstdlib>
#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status AssembleTokensHiddenLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleTokensHiddenLayer not supported.");
}

template <typename T>
Status AssembleTokensHiddenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "AssembleTokensHiddenLayer not supported.");
}
template class AssembleTokensHiddenLayer<float>;
template class AssembleTokensHiddenLayer<float16>;
template class AssembleTokensHiddenLayer<bfloat16>;

}  // namespace ksana_llm
