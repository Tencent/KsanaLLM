/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_mla_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                                 std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: 临时空间
  //     1: token_offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //     4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     5: rotary embedding pos tensor shape [max_token_num], type int64
  //        mrotary embedding pos tensor shape [3, max_token_num], type int64 (only for qwen2_vl)
  //     6: rotary embedding mask tensor shape [max_token_num], type int64
  //     7: flexible_rotary_embedding_pos,
  //     8: flexible_rotary_embedding_mask,
  //     9: dst_flexible_kv_cache_tensor,
  //     10: src_flexible_kv_cache_tensor,
  //     11: dst_flexible_token_idx_tensor,
  //     12: src_flexible_token_idx_tensor,
  //     13: flexible_offset_uint64_tensor,
  //     14: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back(), xx, xx, xx, max_forwarding_tokens]
  //     15: query_layernorm_weight (if not use_qk_norm, value is nullptr)
  //     16: key_layernorm_weight (if not use_qk_norm, value is nullptr)
  //     17: flag_tensor: [use_cache]
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  //     18: kv_cache_base_ptr_tensor: [1 + layer_num * 2]
  //     19: block_table: [batch_size, max_tokens]
  //     20: input_without_prefix_offset_tensor: [batch_size + 1]
#endif
  // output_tensors:
  //     0: flash_mla_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]
  int max_tokens = input_tensors[15].shape[9];
  int batch_size = input_tensors[15].shape[8];
  int layer_block_num = input_tensors[15].shape[2];
  int total_tokens = input_tensors[22].shape[0];
  bool use_cache = input_tensors[18].GetPtr<bool>()[0];

  void** k_list = (input_tensors[2].GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;

  void* q_nope_ptr = input_tensors[22].GetPtr<void>();
  void* q_pe_ptr = input_tensors[23].GetPtr<void>();
  void* compressed_kv_ptr = input_tensors[24].GetPtr<void>();
  void* k_pe_ptr = input_tensors[25].GetPtr<void>();
  void* kv_b_nope_proj_weight = input_tensors[26].GetPtr<void>();
  void* v_head_proj_weight = input_tensors[27].GetPtr<void>();

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = input_tensors[26].weight_scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[27].weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = input_tensors[26].scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[27].scales->GetPtr<void>();
  }

  size_t o_proj_dim = input_tensors[28].shape[1];
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    o_proj_dim = input_tensors[28].shape[0];
  }

  void* prefix_k_buffer = input_tensors[29].GetPtr<void>();
  void* prefix_v_buffer = input_tensors[30].GetPtr<void>();
  void* prefix_o_buffer = input_tensors[31].GetPtr<void>();
  void* prefix_kv_buffer = input_tensors[32].GetPtr<void>();
  void* prefix_k_up_buffer = input_tensors[33].GetPtr<void>();
  void* prefix_v_up_buffer = input_tensors[34].GetPtr<void>();
  void* workspace_buffer = input_tensors[35].GetPtr<void>();

  int64_t kv_cache_block_num = *(input_tensors[19].GetPtr<int64_t>());
  void** layer_kv_cache_ptr = input_tensors[19].GetPtr<void*>() + 1;
  void* k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
  void* v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* block_table_ptr = input_tensors[20].GetPtr<int32_t>();
  int max_blocks_per_seq = input_tensors[20].shape[1];
  size_t* input_without_prefix_offset = input_tensors[21].GetPtr<size_t>();
  int max_forwarding_tokens = input_tensors[15].shape[6];

  // The total length of prefix part.
  int total_prefix_len = input_tensors[15].shape[12];
  std::shared_ptr<Tensor>& work_buffer = this->workspace_buffer_;
  void* fp8_work_buffer = work_buffer == nullptr ? nullptr : work_buffer->GetPtr<void>();

  void* seqlens_q_ptr = input_tensors[4].GetPtr<void>();

  MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
      output_tensors[0].GetPtr<void>(), q_nope_ptr, q_pe_ptr, k_pe_ptr, compressed_kv_ptr, kv_b_nope_proj_weight,
      v_head_proj_weight, kv_b_nope_weight_scale, v_head_weight_scale, o_proj_dim, fp8_work_buffer,
      this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
      input_tensors[6].GetPtr<void>(), input_tensors[7].GetPtr<void>(), workspace_buffer,
      input_tensors[1].GetPtr<void>(), this->attn_scale_, this->rotary_embedding_cuda_, total_tokens, max_tokens,
      batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_, this->kv_lora_rank_,
      this->v_head_dim_, this->num_kv_heads_, this->head_size_, this->stride_size_, this->k_scale_, this->v_scale_,
      this->attn_dp_atp_size_, this->is_causal_, this->rank_, this->block_token_num_, k_list, v_list,
      input_tensors[3].GetPtr<void>(), input_tensors[5].GetPtr<void>(), this->alibi_slopes_, this->layer_index_,
      input_tensors[8].GetPtr<void>(), input_tensors[9].GetPtr<void>(), input_tensors[10].GetPtr<void>(),
      input_tensors[11].GetPtr<void>(), input_tensors[12].GetPtr<void>(), input_tensors[13].GetPtr<void>(),
      input_tensors[14].GetPtr<void>(), input_tensors[10].shape[0], this->layernorm_eps_, this->use_qk_norm_,
      input_tensors[16].GetPtr<void>(), input_tensors[17].GetPtr<void>(), use_cache,
      this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr, block_table_ptr,
      kv_cache_block_num, max_blocks_per_seq, input_without_prefix_offset, max_forwarding_tokens, total_prefix_len,
      seqlens_q_ptr, prefix_k_buffer, prefix_v_buffer, prefix_o_buffer, prefix_kv_buffer, prefix_k_up_buffer,
      prefix_v_up_buffer, this->mm_quant_mode_);

  // 通知 LayerProgressTracker 该层已完成，它会在内部记录 event 并在单独的线程中监控完成情况
  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(this->rank_, this->layer_index_,
                                                                      this->context_->GetComputeStreams()[this->rank_]);

  output_tensors[0].shape = {total_tokens, o_proj_dim};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

using llm_kernels::utils::KVCacheType;
template class FlashMlaAttentionLayer<float, float, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<half, half, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<__nv_bfloat16, __nv_bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class FlashMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
