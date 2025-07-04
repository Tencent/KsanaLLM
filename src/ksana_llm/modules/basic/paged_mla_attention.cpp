/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/paged_mla_attention.h"
#include "ksana_llm/layers/paged_mla_attention_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
PagedMlaAttention<T>::PagedMlaAttention(const size_t layer_idx, bool is_neox, AbsorbWeightsType absorb_type,
                                        const LayerCreationContext<T>& creation_context,
                                        const AttentionCreationConfig& attn_config) {
  attn_dp_group_id_ = creation_context.rank / Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
  uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  uint32_t qk_nope_head_dim = attn_config.model_config.mla_config.qk_nope_head_dim;
  uint32_t q_lora_rank = attn_config.model_config.mla_config.q_lora_rank;
  uint32_t kv_lora_rank = attn_config.model_config.mla_config.kv_lora_rank;
  uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

  paged_mla_attention_layer_ =
      CreateAttentionLayer<T, PagedMlaAttentionLayer>(Singleton<Environment>::GetInstance()->GetKVCacheType());
  // NOTE(karlluo): acsends's image g++ is 9.4.0, it do not support convert
  // from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::any>’ so
  // we use push back to make it work.
  std::vector<std::any> attention_param;
  attention_param.push_back(attn_config.model_config.quant_config.method);
  attention_param.push_back(attn_config.model_config.layernorm_eps);  // for q k layernorm
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
  std::vector<std::any> paged_attention_param = attention_param;

  paged_attention_param.push_back(false);
  // aligned with flash attention
  paged_attention_param.push_back(nullptr);
  paged_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  paged_mla_attention_layer_->Init(paged_attention_param, creation_context.context, creation_context.rank);
  paged_mla_attention_layer_->SetWorkSpaceBuffer(creation_context.matmul_layer_factory->GetWorkspaceBuffer());

  kv_b_nope_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx));
  v_head_proj_weight_ = creation_context.base_weight->GetModelWeights(
      fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx));
  attn_o_proj_weight_ =
      creation_context.base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.o_proj.weight", layer_idx));
  if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
    attn_w_uv_weight_ =
        creation_context.base_weight->GetModelWeights(fmt::format("model.layers.{}.self_attn.w_uv.weight", layer_idx));
  }
}

template <typename T>
Status PagedMlaAttention<T>::Forward(std::vector<Tensor>& output_tensor, ModelInput::input_info& page_input,
                                     std::vector<Tensor>& hidden_buffer_tensors_1, Tensor& kv_cache_buffer_tensor,
                                     const AttentionForwardContext& attn_ctx, Tensor& workspace_buffer,
                                     Tensor& decode_q_buffer_tensor, Tensor& q_rope_buffer_tensor,
                                     Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor) {
  Tensor query_layernorm_weight, key_layernorm_weight;  // qk_norm not supported, use dummy tensor
  STATUS_CHECK_RETURN(paged_mla_attention_layer_->Forward({hidden_buffer_tensors_1[0],
                                                           page_input.input_length,
                                                           page_input.kv_list,
                                                           page_input.kv_cache_offset,
                                                           page_input.rotary_embedding_pos,
                                                           page_input.rotary_embedding_mask,
                                                           kv_cache_buffer_tensor,
                                                           attn_ctx.forward_shape,
                                                           workspace_buffer,       /* workspace */
                                                           query_layernorm_weight, /* for use_qk_norm */
                                                           key_layernorm_weight,   /* for use_qk_norm */
                                                           page_input.layer_kv_cache_ptr,
                                                           page_input.block_table,
                                                           decode_q_buffer_tensor,
                                                           q_rope_buffer_tensor,
                                                           kv_buffer_tensor,
                                                           k_rope_buffer_tensor,
                                                           kv_b_nope_proj_weight_,
                                                           v_head_proj_weight_,
                                                           attn_o_proj_weight_,
                                                           page_input.tile_scheduler_metadata,
                                                           page_input.num_splits,
                                                           page_input.metadata,
                                                           attn_w_uv_weight_},
                                                          output_tensor));

  return Status();
}

template class PagedMlaAttention<float>;
template class PagedMlaAttention<float16>;
#ifdef ENABLE_BFLOAT16
template class PagedMlaAttention<bfloat16>;
#endif

}  // namespace ksana_llm
