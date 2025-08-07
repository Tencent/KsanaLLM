/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class PagedAttention {
 public:
  // Disable a default constructor
  PagedAttention(bool is_neox, const LayerCreationContext& creation_context,
                 const AttentionCreationConfig& attn_config);

  ~PagedAttention();

  // TODO(robertyuan): param after output tensor should be removed
  Status Forward(std::vector<Tensor>& input_tensors, std::shared_ptr<ModelInput>& model_input,
                 std::vector<Tensor>& output_tensors, std::vector<Tensor>& paged_buffer_tensors,
                 Tensor& kv_cache_buffer_tensor, const AttentionForwardContext& forward_context,
                 Tensor query_layernorm_weight, Tensor key_layernorm_weight);

 protected:
  std::shared_ptr<BaseLayer> paged_attention_layer_;
  bool is_cudagraph_enabled_;
  int rank_;
};
}  // namespace ksana_llm
