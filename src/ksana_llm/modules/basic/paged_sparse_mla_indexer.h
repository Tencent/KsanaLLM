/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class PagedSparseMlaIndexer {
 public:
  PagedSparseMlaIndexer(const size_t layer_idx, const LayerCreationContext& creation_context,
                        const AttentionCreationConfig& attn_config, int block_size);

  ~PagedSparseMlaIndexer() = default;

  Status Forward(const std::shared_ptr<ModelInput>& model_input, const ModelInput::input_info& page_input,
                 const AttentionForwardContext& attn_ctx, Tensor& q_indexer_tensor, Tensor& k_indexer_tensor,
                 Tensor& weights_tensor, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> paged_sparse_mla_indexer_layer_;
};

}  // namespace ksana_llm
