/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/split_layer.h"

namespace ksana_llm {

Status SplitLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                        std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}

Status SplitLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("SplitLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "SplitLayer not supported.");
}
}  // namespace ksana_llm
