/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/paged_mla_attention.h"

#include "ksana_llm/layers/paged_mla_attention_layer.h"
#include "ksana_llm/layers/paged_sparse_mla_attention_layer.h"

namespace ksana_llm {

PagedMlaAttention::PagedMlaAttention(const size_t layer_idx, bool is_neox, const LayerCreationContext& creation_context,
                                     const AttentionCreationConfig& attn_config) {
  if (attn_config.model_config.use_dsa) {
    paged_mla_attention_layer_ = std::make_shared<PagedSparseMlaAttentionLayer>();
  } else {
    paged_mla_attention_layer_ = std::make_shared<PagedMlaAttentionLayer>();
  }

  const uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  const uint32_t qk_nope_head_dim = attn_config.model_config.mla_config.qk_nope_head_dim;
  const uint32_t q_lora_rank = attn_config.model_config.mla_config.q_lora_rank;
  const uint32_t kv_lora_rank = attn_config.model_config.mla_config.kv_lora_rank;
  const uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

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
  std::vector<std::any> paged_attention_param = attention_param;
  // bool for is_multi_token_forward
  paged_attention_param.push_back(false);
  // aligned with flash attention
  paged_attention_param.push_back(nullptr);
  paged_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);

  paged_mla_attention_layer_->Init(paged_attention_param, creation_context.runtime_config, creation_context.context,
                                   creation_context.rank);
  paged_mla_attention_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(paged_mla_attention_layer_->GetWorkspaceSize()));
}

Status PagedMlaAttention::Forward(const std::shared_ptr<ModelInput>& model_input,
                                  const ModelInput::input_info& page_input, const AttentionForwardContext& attn_ctx,
                                  std::vector<Tensor>& hidden_buffer_tensors_1, Tensor& decode_q_buffer_tensor,
                                  Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                  Tensor& indices_tensor, std::vector<Tensor>& output_tensors) {
  if (indices_tensor.GetElementNumber() > 0) {
    return paged_mla_attention_layer_->Forward(
        {page_input.rotary_embedding_pos, page_input.rotary_embedding_mask, q_rope_buffer_tensor, k_rope_buffer_tensor,
         decode_q_buffer_tensor, /*q_ptr*/ hidden_buffer_tensors_1[0], kv_buffer_tensor, page_input.kv_list,
         page_input.input_length, page_input.kv_cache_offset, model_input->layer_kv_cache_ptr, page_input.block_table,
         page_input.tile_scheduler_metadata, page_input.num_splits, indices_tensor},
        output_tensors);
  } else {
    return paged_mla_attention_layer_->Forward(
        {hidden_buffer_tensors_1[0], page_input.input_length, page_input.kv_list, page_input.kv_cache_offset,
         page_input.rotary_embedding_pos, page_input.rotary_embedding_mask, model_input->layer_kv_cache_ptr,
         page_input.block_table, decode_q_buffer_tensor, q_rope_buffer_tensor, kv_buffer_tensor, k_rope_buffer_tensor,
         page_input.tile_scheduler_metadata, page_input.num_splits},
        output_tensors);
  }
}

}  // namespace ksana_llm
