/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/modules/basic/bmm.h"

namespace ksana_llm {
class PagedMlaAttention {
 public:
  // Disable a default constructor
  PagedMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                    const AttentionCreationConfig& attn_config);

  ~PagedMlaAttention() = default;

  // TODO(robertyuan): param after output tensor should be removed
  Status Forward(std::vector<Tensor>& output_tensors, const std::shared_ptr<ModelInput>& model_input,
                 const ModelInput::input_info& page_input, std::vector<Tensor>& hidden_buffer_tensors_1,
                 const AttentionForwardContext& attn_ctx, std::vector<Tensor>& workspace_buffer,
                 Tensor& decode_q_buffer_tensor, Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor,
                 Tensor& k_rope_buffer_tensor);

 protected:
  std::shared_ptr<BaseLayer> paged_mla_attention_layer_;
  std::shared_ptr<Bmm> attn_w_uv_bmm_;
};
}  // namespace ksana_llm
