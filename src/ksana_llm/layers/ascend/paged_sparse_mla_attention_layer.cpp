/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_sparse_mla_attention_layer.h"

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

Status PagedSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status PagedSparseMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  KLLM_THROW("PagedSparseMlaAttentionLayer not implement in Ascend.");
}

}  // namespace ksana_llm
