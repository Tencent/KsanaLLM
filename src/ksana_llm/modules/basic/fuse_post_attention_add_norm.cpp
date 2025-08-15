/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/fuse_post_attention_add_norm.h"

namespace ksana_llm {

FusePostAttentionAddNorm::FusePostAttentionAddNorm(const std::string& weight_name, float norm_eps,
                                                   const LayerCreationContext& creation_context) {
  layernorm_layer_ = std::make_shared<Layernorm>(weight_name, norm_eps, creation_context);
  adds_ = std::make_shared<Add>(creation_context);
  add_norm_layer_ = std::make_shared<AddNorm>(weight_name, norm_eps, creation_context);
  if (creation_context.weight_type == TYPE_FP16 || creation_context.weight_type == TYPE_BF16) {
    is_fusion_supported_dtype_ = true;
  }
}

FusePostAttentionAddNorm::~FusePostAttentionAddNorm() {}

Status FusePostAttentionAddNorm::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                         std::vector<Tensor>& residual_buffer) {
  if (is_fusion_supported_dtype_) {
    add_norm_layer_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, hidden_buffer_tensors_0);
  } else {
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
    STATUS_CHECK_RETURN(layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0));
  }
  return Status();
}

}  // namespace ksana_llm
