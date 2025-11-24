/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/modules/basic/linear.h"

namespace ksana_llm {

class FlashMlaAttention {
 public:
  // Disable a default constructor
  FlashMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                    const AttentionCreationConfig& attn_config);

  ~FlashMlaAttention() = default;

  Status Forward(const std::shared_ptr<ModelInput>& model_input, const AttentionForwardContext& attn_ctx,
                 std::vector<Tensor>& k_buffer, std::vector<Tensor>& v_buffer, Tensor& context_q_nope_rope_tensor,
                 Tensor& context_q_nope_tensor, Tensor& context_q_rope_tensor, Tensor& kv_buffer_tensor,
                 Tensor& k_rope_buffer_tensor, Tensor& prefix_kv_buffer_tensor, Tensor& indices_tensor,
                 std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<Linear> kv_b_nope_proj_;
  std::shared_ptr<Linear> v_head_proj_;
  std::shared_ptr<BaseLayer> flash_mla_attention_layer_;

  Tensor kv_b_nope_proj_weight_;
  Tensor v_head_proj_weight_;
};

}  // namespace ksana_llm
