/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {
class PagedMlaAttention {
 public:
  // Disable a default constructor
  PagedMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                    const AttentionCreationConfig& attn_config);

  ~PagedMlaAttention() = default;

  Status Forward(const std::shared_ptr<ModelInput>& model_input, const ModelInput::input_info& page_input,
                 const AttentionForwardContext& attn_ctx, std::vector<Tensor>& hidden_buffer_tensors_1,
                 Tensor& decode_q_buffer_tensor, Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor,
                 Tensor& k_rope_buffer_tensor, Tensor& indices_tensor, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> paged_mla_attention_layer_;
};
}  // namespace ksana_llm
