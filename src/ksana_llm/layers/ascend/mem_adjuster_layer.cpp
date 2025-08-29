/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/mem_adjuster_layer.h"

namespace ksana_llm {

Status MemAdjusterLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("MemAdjusterLayer is not implemented in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "MemAdjusterLayer is not implemented.");
}

Status MemAdjusterLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("MemAdjusterLayer is not implemented in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "MemAdjusterLayer is not implemented.");
}

Status MemAdjusterLayer::ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                          size_t output_n) {
  KLLM_THROW("MemAdjusterLayer is not implemented in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "MemAdjusterLayer is not implemented.");
}

template <typename T>
Status MemAdjusterLayer::ExtractSubMatrixT(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                           size_t output_n) {
  KLLM_THROW("MemAdjusterLayer is not implemented in Ascend.");
  return Status(RET_UNDEFINED_REFERENCE, "MemAdjusterLayer is not implemented.");
}

}  // namespace ksana_llm
