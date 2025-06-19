/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class FlashMlaAttention {
 public:
  // Disable a default constructor
  FlashMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext<T>& creation_context,
                    const AttentionCreationConfig& attn_config);

  ~FlashMlaAttention() = default;

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                 std::vector<Tensor>& hidden_buffer_tensors_1, const AttentionForwardContext& attn_ctx,
                 Tensor& prefill_q_buffer_tensor, Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor,
                 Tensor& k_rope_buffer_tensor, Tensor& prefix_k_buffer_tensor, Tensor& prefix_v_buffer_tensor,
                 Tensor& prefix_o_buffer_tensor, Tensor& prefix_kv_buffer_tensor, Tensor& prefix_k_up_buffer_tensor,
                 Tensor& prefix_v_up_buffer_tensor, Tensor& prefill_buffer_tensor);

 protected:
  std::shared_ptr<BaseLayer> flash_mla_attention_layer_;
  std::shared_ptr<Context> context_;
  int rank_;

  Tensor kv_b_nope_proj_weight_;
  Tensor v_head_proj_weight_;
  Tensor attn_o_proj_weight_;
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     attn   |     attn    |     attn   |     attn    |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  int attn_dp_group_id_ = 0;
};
}  // namespace ksana_llm
