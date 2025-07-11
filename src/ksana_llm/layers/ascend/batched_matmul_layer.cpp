/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/batched_matmul_layer.h"

namespace ksana_llm {

template <typename T>
Status BatchedMatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                   int rank) {
  BaseLayer::Init(parameters, context, rank);
  return Status();
}

template <typename T>
Status BatchedMatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "BatchedMatMulLayer not supported.");
}
template class BatchedMatMulLayer<float>;
template class BatchedMatMulLayer<float16>;
template class BatchedMatMulLayer<bfloat16>;
}  // namespace ksana_llm