/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/flash_attention.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
FlashAttention<T>::FlashAttention(bool is_neox, const LayerCreationContext& creation_context,
                                  const AttentionCreationConfig& attn_config)
    : reuse_prefix_caching_(attn_config.reuse_prefix_caching),
      context_(creation_context.context),
      rank_(creation_context.rank),
      enable_blocked_multi_token_forwarding_kv_(
          creation_context.runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
  uint32_t zero = 0;
  flash_attention_layer_ =
      CreateAttentionLayer<T, FlashAttentionLayer>(creation_context.runtime_config.attn_backend_config.kv_cache_dtype);
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
  std::vector<std::any> flash_attention_param = attention_param;
  flash_attention_param.push_back(true);
  flash_attention_param.push_back(attn_config.mrope_section_ptr);
  flash_attention_param.push_back(attn_config.model_config.enable_qk_pre_norm_before_rotary_pos);
  flash_attention_layer_->Init(flash_attention_param, creation_context.runtime_config, context_, rank_);

  if (attn_config.model_config.rope_scaling_factor_config.type == "mrope") {
    use_mrotary_ = true;
  }
}

template <typename T>
FlashAttention<T>::~FlashAttention() {}

template <typename T>
Status FlashAttention<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                  std::shared_ptr<ModelInput>& model_input,
                                  std::vector<Tensor>& hidden_buffer_tensors_1,
                                  std::vector<Tensor>& shared_buffer_tensors,
                                  const AttentionForwardContext& forward_context, Tensor query_layernorm_weight,
                                  Tensor key_layernorm_weight) {
  if (reuse_prefix_caching_ && !enable_blocked_multi_token_forwarding_kv_) {
    AddAttentionPrefixCache(hidden_buffer_tensors_0, model_input, hidden_buffer_tensors_1, shared_buffer_tensors);
  }

#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(flash_attention_layer_->Forward(
      {hidden_buffer_tensors_0[0],
       model_input->dp_input_offset_uint64_tensor,
       model_input->flash_input.kv_list,
       model_input->dp_input_prefix_uint64_tensor,
       model_input->flash_input.kv_cache_offset,
       use_mrotary_ ? model_input->dp_mrotary_embedding_pos : model_input->flash_input.rotary_embedding_pos,
       model_input->flash_input.rotary_embedding_mask,
       model_input->dp_flexible_rotary_embedding_pos,
       model_input->dp_flexible_rotary_embedding_mask,
       model_input->dp_dst_flexible_kv_cache_tensor,
       model_input->dp_src_flexible_kv_cache_tensor,
       model_input->dp_dst_flexible_token_idx_tensor,
       model_input->dp_src_flexible_token_idx_tensor,
       model_input->dp_flexible_offset_uint64_tensor,
       forward_context.forward_shape,
       query_layernorm_weight, /* for use_qk_norm */
       key_layernorm_weight,   /* for use_qk_norm */
       forward_context.flag_tensor,
       model_input->flash_input.layer_kv_cache_ptr,
       model_input->flash_input.block_table,
       model_input->dp_prefill_q_offset_uint64_tensor},
      hidden_buffer_tensors_1));
#elif defined(ENABLE_ACL)
  // inference on NPU with ATB
  STATUS_CHECK_RETURN(flash_attention_layer_->Forward(
      {hidden_buffer_tensors_0[0], model_input->flash_input.rotary_embedding_pos, model_input->layers_slot_mapping,
       model_input->k_cache_blocks_base, model_input->v_cache_blocks_base, model_input->seq_len_host,
       forward_context.forward_shape, model_input->atb_attention_attr},
      hidden_buffer_tensors_1));
#endif
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

  if (reuse_prefix_caching_ && !enable_blocked_multi_token_forwarding_kv_) {
    RemoveAttentionPrefixCache(hidden_buffer_tensors_0, model_input, hidden_buffer_tensors_1, shared_buffer_tensors);
  }

  return Status();
}

template <typename T>
Status FlashAttention<T>::AddAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                  std::shared_ptr<ModelInput>& model_input,
                                                  std::vector<Tensor>& hidden_buffer_tensors_1,
                                                  std::vector<Tensor>& shared_buffer_tensors) {
  // The input shape for Flash Attention must match the actual token_num, which includes the lengths of both prefix and
  // speculative tokens
  // The mmha_prefix_input reserves space for all tokens. Here, the tokens from mmha_origin_input are copied to their
  // corresponding positions in mmha_prefix_input. For the remaining tokens in mmha_prefix_input, their key-value (KV)
  // pairs will be copied from the KV cache during Flash Attention, while the query (Q) can remain unused.
  const auto& mmha_origin_input = hidden_buffer_tensors_0[0];  // [token_num_without_prefix, hidden]
  auto& mmha_prefix_input = hidden_buffer_tensors_1[0];        // [token_num_with_prefix, hidden]
  size_t total_token_num = 0;                                  // 包含prefix、speculate的实际token总数
  const size_t dtype_size = GetTypeSize(mmha_origin_input.dtype);
  const size_t size_per_token = mmha_origin_input.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input->dp_batch_size; ++idx) {
    const size_t src_offset =
        (model_input->dp_input_offset_list_uint64[idx] - model_input->dp_input_prefix_list_uint64[idx]) *
        size_per_token;
    const size_t input_length =
        model_input->dp_input_offset_list_uint64[idx + 1] - model_input->dp_input_offset_list_uint64[idx];
    const size_t prefix_length =
        model_input->dp_input_prefix_list_uint64[idx + 1] - model_input->dp_input_prefix_list_uint64[idx];
    const size_t copy_size = (input_length - prefix_length) * size_per_token;
    const size_t dst_offset = (model_input->dp_input_offset_list_uint64[idx] + prefix_length) * size_per_token;
    // The single token is located in the latter half; copy it all at once and then exit
    if (idx >= model_input->dp_multi_token_request_num && model_input->dp_single_token_request_num > 0) {
      MemcpyAsync(shared_buffer_tensors[0].template GetPtr<void>(),
                  mmha_origin_input.template GetPtr<void>() + src_offset,
                  copy_size * model_input->dp_single_token_request_num, MEMCPY_DEVICE_TO_DEVICE,
                  context_->GetComputeStreams()[rank_]);
      shared_buffer_tensors[0].shape = {copy_size / dtype_size, model_input->dp_single_token_request_num};
      total_token_num += model_input->dp_single_token_request_num;
      break;
    }
    // Copy multi tokens
    MemcpyAsync(mmha_prefix_input.template GetPtr<void>() + dst_offset,
                mmha_origin_input.template GetPtr<void>() + src_offset, copy_size, MEMCPY_DEVICE_TO_DEVICE,
                context_->GetComputeStreams()[rank_]);
    total_token_num += input_length;
  }
  mmha_prefix_input.shape = {total_token_num, mmha_origin_input.shape[1]};
  mmha_prefix_input.dtype = mmha_origin_input.dtype;
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  return Status();
}

template <typename T>
Status FlashAttention<T>::RemoveAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                     std::shared_ptr<ModelInput>& model_input,
                                                     std::vector<Tensor>& hidden_buffer_tensors_1,
                                                     std::vector<Tensor>& shared_buffer_tensors) {
  // After the completion of MMHA inference, copy the data from the MMHA output results,
  // excluding the Prefix Cache section, and continue with the subsequent inference.
  auto& mmha_prefix_output = hidden_buffer_tensors_0[0];
  auto& mmha_output = hidden_buffer_tensors_1[0];
  auto attention_input_shape = hidden_buffer_tensors_1[0].shape;
  size_t total_token_num_without_prefix = 0;
  size_t dst_offset = 0;
  size_t src_offset = 0;
  size_t dtype_size = GetTypeSize(mmha_prefix_output.dtype);
  size_t size_per_token = mmha_prefix_output.shape[1] * dtype_size;
  for (size_t idx = 0; idx < model_input->dp_multi_token_request_num; ++idx) {
    size_t prefix_length =
        model_input->dp_input_prefix_list_uint64[idx + 1] - model_input->dp_input_prefix_list_uint64[idx];
    size_t input_length =
        model_input->dp_input_offset_list_uint64[idx + 1] - model_input->dp_input_offset_list_uint64[idx];
    src_offset += prefix_length * size_per_token;
    size_t copy_size = size_per_token * (input_length - prefix_length);

    MemcpyAsync(mmha_output.template GetPtr<void>() + dst_offset,
                mmha_prefix_output.template GetPtr<void>() + src_offset, copy_size, MEMCPY_DEVICE_TO_DEVICE,
                context_->GetComputeStreams()[rank_]);
    src_offset += copy_size;
    dst_offset += copy_size;
    total_token_num_without_prefix += input_length - prefix_length;
  }
  mmha_output.shape = {total_token_num_without_prefix, mmha_prefix_output.shape[1]};
  mmha_output.dtype = mmha_prefix_output.dtype;
  if (model_input->dp_single_token_request_num > 0) {
    MemcpyAsync(hidden_buffer_tensors_0[0].template GetPtr<void>() + shared_buffer_tensors[0].GetTotalBytes() /
                                                                         model_input->dp_single_token_request_num *
                                                                         total_token_num_without_prefix,
                shared_buffer_tensors[0].template GetPtr<void>(), shared_buffer_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    hidden_buffer_tensors_0[0].shape = {total_token_num_without_prefix + model_input->dp_single_token_request_num,
                                        attention_input_shape[1]};
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);

  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  return Status();
}

template class FlashAttention<float>;
template class FlashAttention<float16>;
template class FlashAttention<bfloat16>;

}  // namespace ksana_llm
