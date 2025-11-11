/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_sparse_mla_indexer_layer.h"

#include <fmt/core.h>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

Status PagedSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  KLLM_THROW("PagedSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

Status PagedSparseMlaIndexerLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  KLLM_THROW("PagedSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

size_t PagedSparseMlaIndexerLayer::GetWorkspaceSize() {
  KLLM_THROW("PagedSparseMlaIndexerLayer is not supported on NPU/Ascend platform");
}

}  // namespace ksana_llm
