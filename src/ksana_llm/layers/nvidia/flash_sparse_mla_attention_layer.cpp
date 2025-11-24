/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_sparse_mla_attention_layer.h"

#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/flash_mla/flash_sparse_mla.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

Status FlashSparseMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  return AttentionLayer::Init(parameters, runtime_config, context, rank);
}

Status FlashSparseMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
#define DISPATCH_FLASH_SPARSE_MLA_BY_KVTYPE(dtype, kv_cache_dtype, func, ...)                                 \
  switch (kv_cache_dtype) {                                                                                   \
    case TYPE_FP8_DS_MLA:                                                                                     \
      return func<dtype, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla>(__VA_ARGS__);                   \
    default:                                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported KVCacheDtype type: {}.", __PRETTY_FUNCTION__, kv_cache_dtype)); \
  }
  switch (inter_data_type_) {
    case DataType::TYPE_FP16:
      DISPATCH_FLASH_SPARSE_MLA_BY_KVTYPE(float16, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
    case DataType::TYPE_BF16:
      DISPATCH_FLASH_SPARSE_MLA_BY_KVTYPE(bfloat16, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
    default:
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashSparseMlaAttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  int64_t* rotary_embedding_pos_ptr = (input_iter++)->GetPtr<int64_t>();
  int64_t* rotary_embedding_mask_ptr = (input_iter++)->GetPtr<int64_t>();
  const Tensor& q_pe_tensor = *input_iter++;
  void* q_pe_ptr = q_pe_tensor.GetPtr<void>();
  void* k_pe_ptr = (input_iter++)->GetPtr<void>();
  void* q_nope_ptr = (input_iter++)->GetPtr<void>();
  void* q_ptr = (input_iter++)->GetPtr<void>();
  SCALAR_T* compressed_kv_ptr = (input_iter++)->GetPtr<SCALAR_T>();
  const Tensor& kv_list_tensor = *input_iter++;
  size_t* prefix_offsets_ptr = (input_iter++)->GetPtr<size_t>();
  size_t* seqlens_without_prefix_ptr = (input_iter++)->GetPtr<size_t>();
  const Tensor& block_offsets_tensor = *input_iter++;
  int* block_offsets_ptr = block_offsets_tensor.GetPtr<int>();
  const Tensor& layer_kv_cache_ptr_tensor = *input_iter++;
  const Tensor& block_table_tensor = *input_iter++;
  int* block_table_ptr = block_table_tensor.GetPtr<int>();
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  int* tile_scheduler_metadata_ptr = tile_scheduler_metadata_tensor.GetPtr<int>();
  int* num_splits_ptr = (input_iter++)->GetPtr<int>();
  const Tensor& indices_tensor = *input_iter++;
  int* indices_ptr = indices_tensor.GetPtr<int>();

  SCALAR_T* out_ptr = output_tensors[0].GetPtr<SCALAR_T>();

  const size_t token_num = q_pe_tensor.shape[0];
  const size_t batch_size = block_offsets_tensor.shape[0] - 1;
  void** const k_list_ptr = kv_list_tensor.GetPtr<void*>() + this->layer_index_ * kv_list_tensor.shape[1];
  CACHE_T* const kcache_ptr =
      reinterpret_cast<CACHE_T*>(layer_kv_cache_ptr_tensor.GetPtr<void*>()[1 + this->layer_index_ * 2]);
  const int max_num_blocks_per_seq = block_table_tensor.shape[1];
  const int64_t num_blocks = *(layer_kv_cache_ptr_tensor.GetPtr<int64_t>());
  const int num_sm_parts = tile_scheduler_metadata_tensor.shape[0];
  const int topk = indices_tensor.shape[1];

  auto stream = this->context_->GetComputeStreams()[this->rank_].Get();

  KLLM_CHECK_WITH_INFO(this->rotary_embedding_cuda_.has_value(), "rotary_embedding_cuda must be set!");
  // Apply rotary to q_pe [token_num, num_heads_q, qk_rope_head_dim] and k_pe [token_num, num_heads_k, qk_rope_head_dim]
  this->rotary_embedding_cuda_->SetInput(rotary_embedding_pos_ptr, rotary_embedding_mask_ptr, q_pe_ptr, k_pe_ptr,
                                         token_num, stream);
  CUDA_CHECK_LAST_ERROR(this->rotary_embedding_cuda_->Forward<SCALAR_T>());

  // Concat q_nope and q_pe to q [token_num, num_heads_q, qk_nope_head_dim + qk_rope_head_dim]
  constexpr size_t kInnerDimSize = 1;
  const size_t outer_q_dim_size = token_num * this->num_heads_;
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, this->kv_lora_rank_, this->qk_rope_head_dim_, outer_q_dim_size, kInnerDimSize,
                   q_ptr, stream);

  // Store compressed_kv [token_num, num_heads_k, kv_lora_rank] and k_pe [token_num, num_heads_k, qk_rope_head_dim] to
  // the kv cache
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFlashKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      compressed_kv_ptr, reinterpret_cast<SCALAR_T*>(k_pe_ptr), k_list_ptr, k_list_ptr, prefix_offsets_ptr,
      seqlens_without_prefix_ptr, block_offsets_ptr, this->block_token_num_, batch_size, token_num, this->kv_lora_rank_,
      this->qk_rope_head_dim_, this->k_scale_, this->v_scale_, stream));

  // Convert indices [token_num, index_topk] from positions within sequence into positions within block table
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FlashSparseMlaConvertBlockTable(
      indices_ptr, block_table_ptr, seqlens_without_prefix_ptr, token_num, topk, batch_size, max_num_blocks_per_seq,
      this->block_token_num_, stream));

  // Invoke sparse mla kernel to compute the output [token_num, num_heads_q, kv_lora_rank]
  float* const workspace_ptr = workspace_buffer_->GetPtr<float>();
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(q_ptr), kcache_ptr, /*batch_size*/ 1, /*seqlen_q_ori*/ token_num, this->num_heads_,
      /*num_heads_k*/ 1, this->kv_lora_rank_ + this->qk_rope_head_dim_, this->kv_lora_rank_,
      /*k_batch_stride*/ this->block_size_ / this->layer_num_, max_num_blocks_per_seq, num_blocks,
      this->block_token_num_, num_sm_parts, /*seqlens_k_ptr*/ nullptr, /*unused*/ block_table_ptr, this->attn_scale_,
      this->is_causal_, tile_scheduler_metadata_ptr, num_splits_ptr,
      /*is_fp8*/ KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla, indices_ptr, topk, stream, workspace_ptr,
      out_ptr));

  return Status();
}

}  // namespace ksana_llm
