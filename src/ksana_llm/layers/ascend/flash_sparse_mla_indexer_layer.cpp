/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_sparse_mla_indexer_layer.h"

#include <fmt/core.h>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

Status FlashSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("FlashSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

Status FlashSparseMlaIndexerLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  KLLM_THROW("FlashSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() {
  KLLM_THROW("FlashSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

}  // namespace ksana_llm
