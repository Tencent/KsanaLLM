/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/utils/absorb_weights_type.h"

namespace ksana_llm {

template <typename T>
class PagedMlaAttention {
 public:
  // Disable a default constructor
  PagedMlaAttention(const size_t layer_idx, bool is_neox, AbsorbWeightsType absorb_type,
                    const LayerCreationContext& creation_context, const AttentionCreationConfig& attn_config);

  ~PagedMlaAttention() = default;

  // TODO(robertyuan): param after output tensor should be removed
  Status Forward(std::vector<Tensor>& output_tensor, ModelInput::input_info& page_input,
                 std::vector<Tensor>& hidden_buffer_tensors_1, Tensor& kv_cache_buffer_tensor,
                 const AttentionForwardContext& attn_ctx, Tensor& workspace_buffer, Tensor& decode_q_buffer_tensor,
                 Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor);

 protected:
  std::shared_ptr<BaseLayer> paged_mla_attention_layer_;

  Tensor kv_b_nope_proj_weight_;
  Tensor v_head_proj_weight_;
  Tensor attn_o_proj_weight_;

  Tensor attn_w_uv_weight_;
};
}  // namespace ksana_llm
