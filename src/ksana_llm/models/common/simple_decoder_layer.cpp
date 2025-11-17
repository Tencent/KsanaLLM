/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common/simple_decoder_layer.h"

#include <memory>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

SimpleDecoderLayer::SimpleDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias,
                                       LayerCreationContext& creation_context,
                                       ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  adds_ = std::make_shared<Add>(creation_context);

  bool use_qk_norm = model_creation_config.attn_config.use_qk_norm;
  mha_ = std::make_shared<MultiHeadAttention>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                              model_creation_config);
  mlps_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  tp_comm_ = std::make_shared<TpCommunicator>();

  pre_attention_add_norm_ = std::make_shared<FusePreAttentionAddNorm>(
    layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);

  fused_all_reduce_norm_add_post_attn_ = std::make_shared<FusedAllReduceNormAdd>(
    layer_prefix + ".post_attention_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps,
    creation_context, ReduceFuseType::kPostAttn);

  fused_all_reduce_norm_add_pre_attn_ = std::make_shared<FusedAllReduceNormAdd>(
    layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps,
    creation_context, ReduceFuseType::kPreAttn);

  if (creation_context.weight_type == TYPE_FP16 || creation_context.weight_type == TYPE_BF16) {
    use_fused_add_layernorm_ = true;
    // determine if it is the first layer, if it is not the first layer
    // then we need to add the last residual output of the previous layer
    // to make a fused pre-attetion add_norm
    if (layer_idx_ != creation_context.pipeline_config.lower_layer_idx) {
      need_add_residual_before_attn_ = true;
    }
    // determine if it is the last layer in the pipeline, if it is not the last layer
    // then we skip the last layer add operation and add the residual of the last layer
    // to the start of the next layer
    if (layer_idx_ != creation_context.pipeline_config.upper_layer_idx) {
      need_add_residual_after_mlp_ = false;
    }
  }
}

Status SimpleDecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                   ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  if (need_add_residual_before_attn_) {
    STATUS_CHECK_RETURN(fused_all_reduce_norm_add_pre_attn_->Forward(reduce_buffer_tensors, residual_buffer,
      hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context, need_add_residual_before_attn_));
  } else {
    STATUS_CHECK_RETURN(pre_attention_add_norm_->Forward(hidden_buffer_tensors_0, residual_buffer,
      need_add_residual_before_attn_));
  }
  // MultiHeadAttention
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Fused Norm Add
  STATUS_CHECK_RETURN(fused_all_reduce_norm_add_post_attn_->Forward(reduce_buffer_tensors, residual_buffer,
                                                          hidden_buffer_tensors_0, is_multi_token_forward,
                                                          forwarding_context, need_add_residual_before_attn_));
  // Common mlp
  STATUS_CHECK_RETURN(
      mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  if (need_add_residual_after_mlp_) {
     // AllReduce Sum
    tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  }
  return Status();
}

}  // namespace ksana_llm
