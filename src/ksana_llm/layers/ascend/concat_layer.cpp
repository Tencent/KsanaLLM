/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/concat_layer.h"

namespace ksana_llm {

Status ConcatLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  concat_dim = std::any_cast<const size_t>(parameters[0]);
  return Status();
}

Status ConcatLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("ConcatLayer not implement in Ascend.");
  return Status(RET_INFER_FAILED);
}
}  // namespace ksana_llm
