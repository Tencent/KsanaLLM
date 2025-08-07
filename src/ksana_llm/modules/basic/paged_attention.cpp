/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/paged_attention.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

PagedAttention::PagedAttention(bool is_neox, const LayerCreationContext& creation_context,
                               const AttentionCreationConfig& attn_config) {
  paged_attention_layer_ = std::make_shared<PagedAttentionLayer>();

  uint32_t zero = 0;
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
  attention_param.push_back(
      attn_config.model_config.k_scales[attn_config.idx + creation_context.pipeline_config.lower_layer_idx]);
  attention_param.push_back(
      attn_config.model_config.v_scales[attn_config.idx + creation_context.pipeline_config.lower_layer_idx]);
  attention_param.push_back(attn_config.rotary_embedding);
  attention_param.push_back(attn_config.rope_theta);
  // new add for mla
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
  attention_param.push_back(zero);
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
  // NOTE(karlluo): bool for is_multi_token_forward
  paged_attention_param.push_back(false);
  // aligned with flash attention
  paged_attention_param.push_back(nullptr);
  paged_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  paged_attention_layer_->Init(paged_attention_param, creation_context.runtime_config, creation_context.context,
                               creation_context.rank);
}

PagedAttention::~PagedAttention() {}

Status PagedAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& paged_buffer_tensors,
                               Tensor& kv_cache_buffer_tensor, const AttentionForwardContext& forward_context,
                               Tensor query_layernorm_weight, Tensor key_layernorm_weight) {
  // normal page attention only has one input (page_single_input)
  auto& input_info = model_input->page_single_input;

#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(paged_attention_layer_->Forward(
      {hidden_buffer_tensors_0[0], input_info.input_length, input_info.kv_list, input_info.kv_cache_offset,
       input_info.rotary_embedding_pos, input_info.rotary_embedding_mask, kv_cache_buffer_tensor,
       forward_context.forward_shape, paged_buffer_tensors[0], /* workspace */
       query_layernorm_weight,                                 /* for use_qk_norm */
       key_layernorm_weight,                                   /* for use_qk_norm */
       // blocked_multi_token_forwarding_kv
       input_info.layer_kv_cache_ptr, input_info.block_table},
      hidden_buffer_tensors_1));
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
#elif defined(ENABLE_ACL)
  // inference on NPU with ATB
  STATUS_CHECK_RETURN(paged_attention_layer_->Forward(
      {hidden_buffer_tensors_0[0], input_info.rotary_embedding_pos, model_input->layers_slot_mapping,
       model_input->layers_block_table, model_input->k_cache_blocks_base, model_input->v_cache_blocks_base,
       model_input->seq_len_host, forward_context.forward_shape, model_input->atb_attention_attr},
      hidden_buffer_tensors_1));
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
#endif
  return Status();
}

}  // namespace ksana_llm
