/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
  AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
  if (atb_paged_attn_ == nullptr) {
    atb_paged_attn_ = std::make_shared<llm_kernels::ascend::ATBAttention<SCALAR_T>>();
    atb_paged_attn_->Initialize(static_cast<uint32_t>(this->max_batch_size_), this->num_heads_, this->num_kv_heads_,
                                this->head_size_, this->layer_num_, this->layer_index_, this->block_token_num_,
                                context->GetComputeStreams()[rank].Get(), rank, /*is_multi_token_forward*/ false,
                                this->max_position_embeddings_, this->base_);
  }
  return Status();
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  // NOTE(karlluo): for ATB input_tensors:
  //   0: qkv_tensor shape [max_token_num, hidden_units * 3], type same as weight
  //   1: rotary_embedding_pos shape [max_token_num], type int64_t
  //   2: slot_mapping shape [max_token_num], type int32_t
  //   3: blocks_table shape [max_block_num], type int32_t
  //   4: k_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   5: v_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   6: seq_len_host shape [batch_size]
  //   7: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  //   8: atb_attention_attr shape: [2], content: 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  // NOTE(karlluo): output_tensors:
  //   0: paged_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]
  void* output = output_tensors[0].GetPtr<void>();
  void* qkv_tensor = input_tensors[0].GetPtr<void>();
  int64_t hidden_units = this->num_heads_ * this->head_size_;
  int64_t batch_size = input_tensors[7].shape[0];
  output_tensors[0].shape = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(this->rank_))
      ->SetExecuteStream(this->context_->GetComputeStreams()[this->rank_].Get());
  std::vector<int32_t> seq_len_host(batch_size);
  void* rotary_embedding_pos = input_tensors[1].GetPtr<void>();
  void* k_cache = input_tensors[4].GetPtr<void>();
  void* v_cache = input_tensors[5].GetPtr<void>();
  int32_t* seq_len_host_ptr = input_tensors[6].GetPtr<int32_t>();
  std::memcpy(seq_len_host.data(), seq_len_host_ptr, batch_size * sizeof(int32_t));
  uint64_t* atb_attention_attr_ptr = input_tensors[8].GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  uint64_t max_num_blocks_per_query = atb_attention_attr_ptr[1];
  int32_t total_block_num = input_tensors[4].shape[0];
  int32_t total_token_num = input_tensors[0].shape[0];
  int32_t* slot_mapping = input_tensors[2].GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  int32_t* blocks_table =
      input_tensors[3].GetPtr<int32_t>() + this->layer_index_ * batch_size * max_num_blocks_per_query;
  atb_paged_attn_->Forward(output, qkv_tensor, rotary_embedding_pos, reinterpret_cast<void*>(slot_mapping), k_cache,
                           v_cache,
                           /*block_tables*/ reinterpret_cast<void*>(blocks_table),
                           /*max_num_blocks_per_query*/ max_num_blocks_per_query, static_cast<uint32_t>(batch_size),
                           static_cast<uint32_t>(total_token_num), static_cast<uint32_t>(total_block_num),
                           this->block_token_num_, static_cast<uint32_t>(this->layer_index_), seq_len_host.data(),
                           /*is_multi_token_forward*/ false,
                           reinterpret_cast<atb::Context*>(GetRuntimeContext(this->rank_)), GetWorkSpaceFunc());
  return Status();
}

using llm_kernels::utils::KVCacheType;
template class PagedAttentionLayer<float, float, KVCacheType::kAuto>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedAttentionLayer<float16, float16, KVCacheType::kAuto>;
template class PagedAttentionLayer<float16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<float16, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedAttentionLayer<bfloat16, bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class PagedAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
