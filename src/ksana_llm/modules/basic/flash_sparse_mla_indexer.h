/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class FlashSparseMlaIndexer {
 public:
  // Disable a default constructor
  FlashSparseMlaIndexer(const size_t layer_idx, const LayerCreationContext& creation_context,
                        const AttentionCreationConfig& attn_config, int block_size);

  ~FlashSparseMlaIndexer() = default;

  Status Forward(const std::shared_ptr<ModelInput>& model_input, const AttentionForwardContext& attn_ctx,
                 Tensor& q_indexer_tensor, Tensor& k_indexer_tensor, Tensor& weights_tensor,
                 std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> flash_sparse_mla_indexer_layer_;
  std::shared_ptr<Context> context_;
  int rank_;
};

}  // namespace ksana_llm
