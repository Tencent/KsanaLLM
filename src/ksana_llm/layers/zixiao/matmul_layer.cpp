/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "MatMulLayer not supported.");
}

template <typename T>
Status MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "MatMulLayer not supported.");
}

template class MatMulLayer<float>;
template class MatMulLayer<float16>;
template class MatMulLayer<bfloat16>;
}  // namespace ksana_llm
