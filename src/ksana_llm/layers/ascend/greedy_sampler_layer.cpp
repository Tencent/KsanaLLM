/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/greedy_sampler_layer.h"

namespace ksana_llm {

Status GreedySamplerLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status GreedySamplerLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "GreedySamplerLayer not supported.");
}

}  // namespace ksana_llm
