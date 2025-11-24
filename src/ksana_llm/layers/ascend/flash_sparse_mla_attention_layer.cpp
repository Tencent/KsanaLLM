/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_sparse_mla_attention_layer.h"

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

Status FlashSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status FlashSparseMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  KLLM_THROW("FlashSparseMlaAttentionLayer not implement in Ascend.");
}

}  // namespace ksana_llm
