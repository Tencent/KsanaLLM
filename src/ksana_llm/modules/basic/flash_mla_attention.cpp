/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/flash_mla_attention.h"

#include "ksana_llm/layers/flash_mla_attention_layer.h"

namespace ksana_llm {

FlashMlaAttention::FlashMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                                     const AttentionCreationConfig& attn_config) {
  uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  uint32_t qk_nope_head_dim = attn_config.model_config.mla_config.qk_nope_head_dim;
  uint32_t q_lora_rank = attn_config.model_config.mla_config.q_lora_rank;
  uint32_t kv_lora_rank = attn_config.model_config.mla_config.kv_lora_rank;
  uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

  flash_mla_attention_layer_ = std::make_shared<FlashMlaAttentionLayer>();

  // NOTE(karlluo): acsends's image g++ is 9.4.0, it do not support convert
  // from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::any>’ so
  // we use push back to make it work.
  std::vector<std::any> attention_param;
  attention_param.push_back(attn_config.model_config.quant_config.method);  // for quant method
  attention_param.push_back(attn_config.model_config.layernorm_eps);        // for q k layernorm
  attention_param.push_back(attn_config.model_config.use_qk_norm);
  attention_param.push_back(attn_config.idx);
  attention_param.push_back(attn_config.layer_num_on_node);
  attention_param.push_back(attn_config.max_position_embeddings);
  attention_param.push_back(attn_config.head_num_per_tp);
  attention_param.push_back(attn_config.num_kv_heads_per_tp);
  attention_param.push_back(attn_config.size_per_head);
  attention_param.push_back(attn_config.stride_size);
  attention_param.push_back(attn_config.tensor_para_size);
  attention_param.push_back(attn_config.kv_cache_dtype);
  attention_param.push_back(attn_config.model_config.k_scales[layer_idx]);
  attention_param.push_back(attn_config.model_config.v_scales[layer_idx]);
  attention_param.push_back(attn_config.rotary_embedding);
  attention_param.push_back(attn_config.rope_theta);
  // new add for mla
  attention_param.push_back(qk_rope_head_dim);
  attention_param.push_back(qk_nope_head_dim);
  attention_param.push_back(q_lora_rank);
  attention_param.push_back(kv_lora_rank);
  attention_param.push_back(v_head_dim);
  // end new add for mla
  attention_param.push_back(is_neox);
  attention_param.push_back(attn_config.position_encoding);
  attention_param.push_back(attn_config.cos_sin_cache_ptr);
  attention_param.push_back(attn_config.model_config.rope_scaling_factor_config);
  attention_param.push_back(attn_config.max_batch_size);
  // add for applying temperature tuning
  attention_param.push_back(attn_config.model_config.attn_temperature_tuning);
  attention_param.push_back(attn_config.model_config.attn_scale);
  attention_param.push_back(attn_config.model_config.floor_scale);
  // end for applying temperature tuning
  std::vector<std::any> flash_attention_param = attention_param;
  // NOTE(karlluo): bool for
  // is_multi_token_forward
  flash_attention_param.push_back(true);
  flash_attention_param.push_back(attn_config.mrope_section_ptr);
  flash_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);

  flash_mla_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, creation_context.context,
                                   creation_context.rank);

  flash_mla_attention_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_mla_attention_layer_->GetWorkspaceSize()));

  // Initialize proj module
  kv_b_nope_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx),
                                             creation_context, attn_config.model_config.quant_config.backend);
  // `kv_b_nope_proj_` and `v_head_proj_` have the same input and share the quantization results
  // We skip the quantization in `v_head_proj_`
  v_head_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx),
                                          creation_context, attn_config.model_config.quant_config.backend,
                                          /*skip_quant*/ true);

  kv_b_nope_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx));
  v_head_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx));
}

Status FlashMlaAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                  const std::shared_ptr<ModelInput>& model_input, std::vector<Tensor>& k_buffer,
                                  std::vector<Tensor>& v_buffer, const AttentionForwardContext& attn_ctx,
                                  Tensor& q_nope_rope_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                  Tensor& prefix_kv_buffer_tensor, std::vector<Tensor>& output_tensors) {
  // When prefix cache hits, the following proj must be applied after fetching the prefix from kv cache
  if (/*dp_total_prefix_tokens*/ attn_ctx.forward_shape.shape[11] == 0) {
    // The quantized result of kv_buffer is stored in the workspace_buffer shared across layers,
    // and immediately required by v_head_proj, so no layer operations can be inserted between them
    STATUS_CHECK_RETURN(kv_b_nope_proj_->Forward(kv_buffer_tensor, output_tensors));
    STATUS_CHECK_RETURN(v_head_proj_->Forward(kv_buffer_tensor, v_buffer));
  }

  STATUS_CHECK_RETURN(flash_mla_attention_layer_->Forward({model_input->dp_input_offset_uint64_tensor,
                                                           model_input->dp_input_offset_int32_tensor,
                                                           model_input->flash_input.kv_list,
                                                           model_input->dp_input_prefix_uint64_tensor,
                                                           model_input->dp_prefill_q_offset_uint64_tensor,
                                                           model_input->dp_prefill_q_offset_int32_tensor,
                                                           model_input->flash_input.kv_cache_offset,
                                                           model_input->flash_input.rotary_embedding_pos,
                                                           model_input->flash_input.rotary_embedding_mask,
                                                           model_input->dp_flexible_rotary_embedding_pos,
                                                           model_input->dp_flexible_rotary_embedding_mask,
                                                           model_input->dp_dst_flexible_kv_cache_tensor,
                                                           model_input->dp_src_flexible_kv_cache_tensor,
                                                           model_input->dp_dst_flexible_token_idx_tensor,
                                                           model_input->dp_src_flexible_token_idx_tensor,
                                                           model_input->dp_flexible_offset_uint64_tensor,
                                                           attn_ctx.forward_shape,
                                                           model_input->flash_input.layer_kv_cache_ptr,
                                                           model_input->flash_input.block_table,
                                                           q_nope_rope_tensor,
                                                           kv_buffer_tensor,
                                                           k_rope_buffer_tensor,
                                                           kv_b_nope_proj_weight_,
                                                           v_head_proj_weight_,
                                                           prefix_kv_buffer_tensor,
                                                           k_buffer[0],
                                                           v_buffer[0]},
                                                          output_tensors));

  return Status();
}
}  // namespace ksana_llm
