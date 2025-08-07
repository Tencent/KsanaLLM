/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_mla_attention_layer.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

Status PagedMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status PagedMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("PagedMlaAttentionLayer not implement in Ascend.");
  return Status();
}


}  // namespace ksana_llm
