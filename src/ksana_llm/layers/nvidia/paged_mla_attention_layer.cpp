/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_mla_attention_layer.h"

#include <thread>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer<SCALAR_T>::Init(parameters, context, rank);

  // index 25 is max_batch_size in PagedMlaAttention, disgusting code, be careful.
  const size_t max_batch_size = std::any_cast<const size_t>(parameters[25]);
  SetMlaMetadataKernelAttribute(max_batch_size, context->GetComputeStreams()[rank].Get());

  return Status();
}

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  const Tensor& hidden_buffer1 = *input_iter++;
  const Tensor& kv_seq_len = *input_iter++;  // kv seq len (len of forwarding_tokens)
  const Tensor& kv_list = *input_iter++;
  const Tensor& cache_offset = *input_iter++;
  const Tensor& rotary_embedding_pos = *input_iter++;
  const Tensor& rotary_embedding_mask = *input_iter++;
  const Tensor& workspace = *input_iter++;
  const Tensor& forward_shape = *input_iter++;
  const Tensor& qkv_workspace = *input_iter++;
  const Tensor& query_norm_weight = *input_iter++;  // not supported, just an empty tensor
  const Tensor& key_norm_weight = *input_iter++;    // not supported, just an empty tensor
  const Tensor& layer_kv_cache = *input_iter++;
  const Tensor& block_table = *input_iter++;
  const Tensor& q_nope_tensor = *input_iter++;
  const Tensor& q_pe_tensor = *input_iter++;
  const Tensor& compressed_kv_tensor = *input_iter++;
  const Tensor& k_pe_tensor = *input_iter++;
  const Tensor& kv_b_nope_proj_weight = *input_iter++;
  const Tensor& v_head_proj_weight = *input_iter++;
  const Tensor& o_proj_weight = *input_iter++;
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  const Tensor& num_splits_tensor = *input_iter++;
  const Tensor& metadata = *input_iter++;
  const Tensor& w_uv_weight = *input_iter++;

  Tensor& out = output_tensors[0];

  const size_t skip_tokens_num = metadata.GetPtr<size_t>()[0];  // 在input_ids中的offset
  const size_t q_seq_len = metadata.GetPtr<size_t>()[1];

  const size_t batch_size = kv_seq_len.shape[0];
  const size_t total_tokens = batch_size * q_seq_len;

  void** const k_list_base = kv_list.GetPtr<void*>();
  const size_t layer_block_num = kv_list.shape[1] * 0.5;  // shape: [layer_num_on_node, total_block_num * 2]
  void** const k_list = k_list_base + static_cast<size_t>(this->layer_index_ * layer_block_num * 2);
  void** const v_list = k_list + layer_block_num;
  int64_t kv_cache_block_num = *(layer_kv_cache.GetPtr<int64_t>());
  void** const layer_kv_cache_ptr = layer_kv_cache.GetPtr<void*>() + 1;
  void* const k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];  // block中每层layer的起始地址
  void* v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* const block_table_ptr =
      block_table.GetPtr<int32_t>();                    // block id，加上layer_kv_cache_ptr后就是对应的cache block
  const int max_blocks_per_seq = block_table.shape[1];  // shape: [bs, max_num_blocks_per_query]
  // for mla
  auto skipped_q_pe_ptr =
      q_pe_tensor.GetPtr<void>() + skip_tokens_num * (q_pe_tensor.GetTotalBytes() / q_pe_tensor.shape[0]);
  auto skipped_compressed_kv_ptr =
      compressed_kv_tensor.GetPtr<void>() +
      skip_tokens_num * (compressed_kv_tensor.GetTotalBytes() / compressed_kv_tensor.shape[0]);

  auto skipped_k_pe_ptr =
      k_pe_tensor.GetPtr<void>() + skip_tokens_num * (k_pe_tensor.GetTotalBytes() / k_pe_tensor.shape[0]);

  const size_t batch_input_ids_len = q_pe_tensor.shape[0];
  const size_t batch_tail_tokens = batch_input_ids_len - skip_tokens_num - total_tokens;

  void* const kv_b_nope_proj_weight_ptr = kv_b_nope_proj_weight.GetPtr<void>();
  void* const v_head_proj_weight_ptr = v_head_proj_weight.GetPtr<void>();

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = input_tensors[17].weight_scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[18].weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = input_tensors[17].scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[18].scales->GetPtr<void>();
  }

  const size_t o_proj_dim =
      this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3 ? o_proj_weight.shape[0] : o_proj_weight.shape[1];

  auto skipped_hidden_buffer1_ptr =
      hidden_buffer1.GetPtr<void>() + skip_tokens_num * (hidden_buffer1.GetTotalBytes() / hidden_buffer1.shape[0]);
  const size_t o_proj_k_dim = this->v_head_dim_ * this->num_heads_;
  auto skipped_output_ptr = out.GetPtr<void>() + skip_tokens_num * o_proj_k_dim * out.GetDTypeSize();

  std::shared_ptr<Tensor>& work_buffer = this->workspace_buffer_;
  void* fp8_work_buffer = work_buffer == nullptr ? nullptr : work_buffer->GetPtr<void>();

  void* const w_uv_weight_ptr = w_uv_weight.GetPtr<void>();

  if (w_uv_weight_ptr) {  // Absorb
    if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
      // 量化时存储convert table的起始位置
      v_cache_ptr = layer_kv_cache_ptr[0] +
                    kv_cache_block_num * this->block_token_num_ * (this->kv_lora_rank_ + this->qk_rope_head_dim_);
      KLLM_LOG_DEBUG << "v_cache_ptr " << v_cache_ptr << " offset = "
                     << kv_cache_block_num * this->block_token_num_ * (this->kv_lora_rank_ + this->qk_rope_head_dim_)
                     << " kv_size = " << (this->kv_lora_rank_ + this->qk_rope_head_dim_);
      kv_cache_block_num = kv_cache_block_num / this->layer_num_;
      KLLM_LOG_DEBUG << "kv_cache_block_num " << kv_cache_block_num;
    }
    InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
        skipped_hidden_buffer1_ptr, skipped_output_ptr, q_nope_tensor.GetPtr<void>(), skipped_q_pe_ptr,
        skipped_compressed_kv_ptr, skipped_k_pe_ptr, w_uv_weight_ptr, o_proj_dim, fp8_work_buffer,
        this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
        k_list, kv_seq_len.GetPtr<void>(), this->context_->GetComputeStreams()[this->rank_].Get(),
        cache_offset.GetPtr<void>(), this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_,
        this->kv_lora_rank_, this->v_head_dim_, this->num_kv_heads_, this->block_token_num_, this->k_scale_,
        this->v_scale_, batch_size, rotary_embedding_pos.GetPtr<void>(), rotary_embedding_mask.GetPtr<void>(),
        total_tokens, this->attn_scale_, this->rotary_embedding_cuda_, tile_scheduler_metadata_tensor.GetPtr<void>(),
        num_splits_tensor.GetPtr<void>(), this->rank_, qkv_workspace.GetPtr<void>(), k_cache_ptr, v_cache_ptr,
        block_table_ptr, kv_cache_block_num, max_blocks_per_seq, q_seq_len, batch_tail_tokens);
    return Status();
  }

  const size_t max_tokens = forward_shape.shape[11];  // dp_single_token_request_max_tokens
  InvokeMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
      skipped_hidden_buffer1_ptr, skipped_output_ptr, q_nope_tensor.GetPtr<void>(), skipped_q_pe_ptr,
      skipped_compressed_kv_ptr, skipped_k_pe_ptr, kv_b_nope_proj_weight_ptr, v_head_proj_weight_ptr,
      kv_b_nope_weight_scale, v_head_weight_scale, o_proj_dim, fp8_work_buffer,
      this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
      k_list, v_list, kv_seq_len.GetPtr<void>(), max_tokens, this->context_->GetComputeStreams()[this->rank_].Get(),
      cache_offset.GetPtr<void>(), batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_,
      this->kv_lora_rank_, this->v_head_dim_, this->head_size_, this->num_kv_heads_, this->stride_size_,
      this->block_token_num_, this->k_scale_, this->v_scale_, batch_size, rotary_embedding_pos.GetPtr<void>(),
      rotary_embedding_mask.GetPtr<void>(), total_tokens, this->attn_scale_, this->rotary_embedding_cuda_,
      workspace.GetPtr<void>(), this->layernorm_eps_, this->use_qk_norm_, query_norm_weight.GetPtr<void>(),
      key_norm_weight.GetPtr<void>(), workspace.GetTotalBytes(), this->rank_, this->alibi_slopes_,
      qkv_workspace.GetPtr<void>(), k_cache_ptr, v_cache_ptr, block_table_ptr, kv_cache_block_num, max_blocks_per_seq,
      this->mm_quant_mode_, q_seq_len);

  return Status();
}

using llm_kernels::utils::KVCacheType;
template class PagedMlaAttentionLayer<float, float, KVCacheType::kAuto>;
template class PagedMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedMlaAttentionLayer<half, half, KVCacheType::kAuto>;
template class PagedMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedMlaAttentionLayer<__nv_bfloat16, __nv_bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class PagedMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
