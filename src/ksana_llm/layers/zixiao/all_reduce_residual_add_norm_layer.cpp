/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"

namespace ksana_llm {

Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
  std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AllReduceResidualAddNormLayer not supported.");
}

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "AllReduceResidualAddNormLayer not supported.");
}

}  // namespace ksana_llm
