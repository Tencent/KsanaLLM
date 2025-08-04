/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/batched_matmul_layer.h"

namespace ksana_llm {

Status BatchedMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                   std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status BatchedMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "BatchedMatMulLayer not supported.");
}
}  // namespace ksana_llm
