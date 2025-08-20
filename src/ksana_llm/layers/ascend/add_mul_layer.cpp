/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_mul_layer.h"

namespace ksana_llm {

Status AddMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AddMul operation is not supported on Ascend backend");
}

Status AddMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("AddMulLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "AddMulLayer not supported.");
}

template <typename T>
Status AddMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("AddMulLayer not implement in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "AddMulLayer not supported.");
}

}  // namespace ksana_llm