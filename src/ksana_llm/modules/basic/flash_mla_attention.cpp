/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/flash_mla_attention.h"
#include "ksana_llm/layers/flash_mla_attention_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
FlashMlaAttention<T>::FlashMlaAttention(const size_t layer_idx, bool is_neox,
                                        const LayerCreationContext<T>& creation_context,
                                        const AttentionCreationConfig& attn_config)
    : context_(creation_context.context), rank_(creation_context.rank) {
  uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  uint32_t qk_nope_head_dim = attn_config.model_config.mla_config.qk_nope_head_dim;
  uint32_t q_lora_rank = attn_config.model_config.mla_config.q_lora_rank;
  uint32_t kv_lora_rank = attn_config.model_config.mla_config.kv_lora_rank;
  uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

  flash_mla_attention_layer_ = CreateAttentionLayer<T, FlashMlaAttentionLayer>(
      creation_context.runtime_config.attn_backend_config.kv_cache_dtype);
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
  attention_param.push_back(attn_config.data_type);
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

  flash_mla_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, context_, rank_);

  flash_mla_attention_layer_->SetWorkSpaceBuffer(creation_context.matmul_layer_factory->GetWorkspaceBuffer());

  kv_b_nope_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx));
  v_head_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx));
  attn_o_proj_weight_ =
      creation_context.base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
}

template <typename T>
Status FlashMlaAttention<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                     std::shared_ptr<ModelInput>& model_input,
                                     std::vector<Tensor>& hidden_buffer_tensors_1,
                                     const AttentionForwardContext& attn_ctx, Tensor& prefill_q_buffer_tensor,
                                     Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor,
                                     Tensor& k_rope_buffer_tensor, Tensor& prefix_k_buffer_tensor,
                                     Tensor& prefix_v_buffer_tensor, Tensor& prefix_kv_buffer_tensor,
                                     Tensor& prefix_k_up_buffer_tensor, Tensor& prefix_v_up_buffer_tensor,
                                     std::vector<Tensor>& output_tensors) {
  Tensor query_layernorm_weight, key_layernorm_weight;  // qk_norm not supported, use dummy tensor
  STATUS_CHECK_RETURN(flash_mla_attention_layer_->Forward({hidden_buffer_tensors_0[0],
                                                           model_input->dp_input_offset_uint64_tensor,
                                                           model_input->flash_input.kv_list,
                                                           model_input->dp_input_prefix_uint64_tensor,
                                                           model_input->dp_prefill_q_offset_uint64_tensor,
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
                                                           query_layernorm_weight, /* for use_qk_norm */
                                                           key_layernorm_weight,   /* for use_qk_norm */
                                                           attn_ctx.flag_tensor,
                                                           model_input->flash_input.layer_kv_cache_ptr,
                                                           model_input->flash_input.block_table,
                                                           model_input->dp_input_without_prefix_uint64_tensor,
                                                           prefill_q_buffer_tensor,
                                                           q_rope_buffer_tensor,
                                                           kv_buffer_tensor,
                                                           k_rope_buffer_tensor,
                                                           kv_b_nope_proj_weight_,
                                                           v_head_proj_weight_,
                                                           attn_o_proj_weight_,
                                                           prefix_k_buffer_tensor,
                                                           prefix_v_buffer_tensor,
                                                           prefix_kv_buffer_tensor,
                                                           prefix_k_up_buffer_tensor,
                                                           prefix_v_up_buffer_tensor,
                                                           hidden_buffer_tensors_1[0]},
                                                          output_tensors));
  return Status();
}

template class FlashMlaAttention<float>;
template class FlashMlaAttention<float16>;
template class FlashMlaAttention<bfloat16>;

}  // namespace ksana_llm
