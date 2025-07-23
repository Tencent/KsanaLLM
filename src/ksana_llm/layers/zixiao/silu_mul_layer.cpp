/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"

namespace ksana_llm {

template <typename T>
Status SiluMulLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}

template <typename T>
Status SiluMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "SiluMulLayer not supported.");
}
template class SiluMulLayer<float>;
template class SiluMulLayer<float16>;
template class SiluMulLayer<bfloat16>;
}  // namespace ksana_llm
