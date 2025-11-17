/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/common_attention.h"

namespace ksana_llm {

CommonAttention::CommonAttention(int layer_idx, bool is_neox, bool use_qk_norm, LayerCreationContext& creation_context,
                                 ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  attn_o_projs_ = std::make_shared<Linear>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                           model_creation_config.attn_config.model_config.quant_config.backend);

  if (use_qk_norm || model_creation_config.attn_config.model_config.enable_qk_pre_norm_before_rotary_pos) {
    query_layernorm_weight_ =
        creation_context.base_weight->GetModelWeights(layer_prefix + ".self_attn.query_layernorm.weight");
    key_layernorm_weight_ =
        creation_context.base_weight->GetModelWeights(layer_prefix + ".self_attn.key_layernorm.weight");
  }

  model_creation_config.attn_config.idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  if (layer_idx >= static_cast<int>(model_creation_config.attn_config.model_config.num_layer)) {
    model_creation_config.attn_config.idx = creation_context.pipeline_config.upper_layer_idx -
                                            creation_context.pipeline_config.lower_layer_idx + layer_idx -
                                            model_creation_config.attn_config.model_config.num_layer + 1;
  }
  flash_attentions_ = std::make_shared<FlashAttention>(is_neox, creation_context, model_creation_config.attn_config);
  paged_attentions_ = std::make_shared<PagedAttention>(is_neox, creation_context, model_creation_config.attn_config);
}

Status CommonAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  auto& shared_buffer_tensors = reduce_buffer_tensors;
  auto& paged_buffer_tensors = reduce_buffer_tensors;

  // MMHA Flash/Paged Attention
  if (!forwarding_context.GetModelInput()->is_cudagraph_capture_request && layer_idx_ == 0) {
    // only need sync in the first layer
    StreamWaitEvent(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()],
                    forwarding_context.GetModelInput()->kvcache_offset_event);
    StreamWaitEvent(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()],
                    forwarding_context.GetModelInput()->rotary_embedding_event);
  }
  if (forwarding_context.GetModelInput()->multi_token_request_num) {
    flash_attentions_->Forward(hidden_buffer_tensors_0, forwarding_context.GetModelInput(), hidden_buffer_tensors_1,
                               shared_buffer_tensors, forwarding_context.GetAttentionForwardContext(),
                               query_layernorm_weight_, key_layernorm_weight_);
    if (forwarding_context.GetModelInput()->single_token_request_num) {
      std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
    }
  }

  if (forwarding_context.GetModelInput()->single_token_request_num) {
    CREATE_BUFFER_SCOPE(kv_cache_buffer_tensors, forwarding_context.GetForwardingBuffers()->kv_cache_buffer);
    paged_attentions_->Forward(hidden_buffer_tensors_0, forwarding_context.GetModelInput(), hidden_buffer_tensors_1,
                               paged_buffer_tensors, kv_cache_buffer_tensors[0],
                               forwarding_context.GetAttentionForwardContext(), query_layernorm_weight_,
                               key_layernorm_weight_);
  }

  // Attn o_proj MatMul
  if (forwarding_context.GetModelCommunicator()) {
    // Put output to `reduce_buffer_tensors` to ensure that the input for custom reduce sum is
    // always in `reduce_buffer_tensors`.
    STATUS_CHECK_RETURN(attn_o_projs_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(attn_o_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

  return Status();
}

}  // namespace ksana_llm
