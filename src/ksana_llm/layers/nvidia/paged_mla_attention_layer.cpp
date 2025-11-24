/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_mla_attention_layer.h"

#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"
#include "ksana_llm//utils/memory_utils.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

// Adapted from
// [DeepSeek-V3 Project]
// https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/attention/backends/triton_mla.py#L698
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeAbsorbMlaPagedAttention(void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr,
                                   void* compressed_kv_ptr, void* k_pe_ptr, void** key_cache_ptrs,
                                   void* context_lens_ptr, cudaStream_t stream, void* cache_offsets_ptr, int num_heads,
                                   int qk_rope_head_dim, int kv_lora_rank, int block_size, float k_scale, float v_scale,
                                   int batch_size, void* rotary_embedding_pos, void* rotary_embedding_mask,
                                   int total_tokens, float attn_scale,
                                   std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,
                                   void* tile_scheduler_metadata_ptr, void* num_splits_ptr, int rank, void* k_cache_ptr,
                                   int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                                   int q_seq_len) {
  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask), q_pe_ptr, k_pe_ptr, total_tokens,
                                    stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
  }

  void* q_concat_ptr = hidden_buffer_1;
  constexpr size_t kInnerDimSize = 1;
  const size_t outer_q_dim_size = total_tokens * num_heads;
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_q_dim_size, kInnerDimSize, q_concat_ptr,
                   stream);

  void* qkv_workspace =
      hidden_buffer_1 + outer_q_dim_size * (kv_lora_rank + qk_rope_head_dim) * sizeof(SCALAR_T);

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaPagedKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(compressed_kv_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), key_cache_ptrs,
      reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size, batch_size,
      q_seq_len, kv_lora_rank, qk_rope_head_dim, k_scale, stream));

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
    KLLM_THROW("Flash MLA not support fp8_e5m2 KV Cache. Please use fp8_e4m3.");
  } else if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    KLLM_LOG_DEBUG << "FP8 kv cache and flash mla enabled, using FP8 inference, quantizing q tensor.";
    const float q_scale = k_scale;
    // Quant q_concat and store into qkv_workspace (cannot be done in-place)
    void* const quant_q_tensor_ptr = qkv_workspace;
    qkv_workspace += outer_q_dim_size * (kv_lora_rank + qk_rope_head_dim) * sizeof(CACHE_T);
    llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        /*q_src*/ reinterpret_cast<SCALAR_T*>(q_concat_ptr), /*q_dst*/ reinterpret_cast<CACHE_T*>(quant_q_tensor_ptr),
        batch_size * q_seq_len, num_heads, kv_lora_rank + qk_rope_head_dim,
        num_heads * (kv_lora_rank + qk_rope_head_dim), q_scale, stream);
    q_concat_ptr = quant_q_tensor_ptr;
  }

  // Flash mla accepts CACHE_T type input. If KV_DTYPE is auto, SCALAR_T equals CACHE_T.
  // If KV_DTYPE is e4m3, flash mla calculates at fp8 precision and outputs at bf16 precision.
  llm_kernels::nvidia::InvokeFlashMla<SCALAR_T, CACHE_T, KV_DTYPE>(
      static_cast<CACHE_T*>(q_concat_ptr), static_cast<CACHE_T*>(k_cache_ptr), q_seq_len, attn_scale, block_table_ptr,
      context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace,
      /*attn_out*/ output_ptr, batch_size, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, k_scale, v_scale,
      max_blocks_per_seq, rank, kv_cache_block_num, stream);
}

#define RUN_ABSORB_MLA_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                   \
  template void InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                           \
      void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr,             \
      void* k_pe_ptr, void** key_cache_ptrs, void* context_lens_ptr, cudaStream_t stream, void* cache_offsets_ptr,    \
      int num_heads, int qk_rope_head_dim, int kv_lora_rank, int block_size, float k_scale, float v_scale,            \
      int batch_size, void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,    \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,                                 \
      void* tile_scheduler_metadata_ptr, void* num_splits_ptr, int rank, void* k_cache_ptr, int32_t* block_table_ptr, \
      int64_t kv_cache_block_num, int max_blocks_per_seq, int q_seq_len)
RUN_ABSORB_MLA_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_ABSORB_MLA_PAGED_ATTENTION

Status PagedMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);

  // index 25 is max_batch_size in PagedMlaAttention, disgusting code, be careful.
  const size_t max_batch_size = std::any_cast<const size_t>(parameters[25]);
  llm_kernels::nvidia::SetFlashMlaAttribute(max_batch_size, context->GetComputeStreams()[rank].Get());

  return Status();
}

Status PagedMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
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
Status PagedMlaAttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  const Tensor& hidden_buffer_1 = *input_iter++;
  const Tensor& kv_seq_len = *input_iter++;  // kv seq len (len of forwarding_tokens)
  const Tensor& kv_list = *input_iter++;
  const Tensor& cache_offset = *input_iter++;
  const Tensor& rotary_embedding_pos = *input_iter++;
  const Tensor& rotary_embedding_mask = *input_iter++;
  const Tensor& layer_kv_cache = *input_iter++;
  const Tensor& block_table = *input_iter++;
  const Tensor& q_nope_tensor = *input_iter++;
  const Tensor& q_pe_tensor = *input_iter++;
  const Tensor& compressed_kv_tensor = *input_iter++;
  const Tensor& k_pe_tensor = *input_iter++;
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  const Tensor& num_splits_tensor = *input_iter++;

  Tensor& output = output_tensors[0];

  const size_t batch_size = kv_seq_len.shape[0];
  const size_t total_tokens = k_pe_tensor.shape[0];
  const size_t q_seq_len = total_tokens / batch_size;

  void** const k_list = kv_list.GetPtr<void*>() + this->layer_index_ * kv_list.shape[1];
  const int64_t kv_cache_block_num = *(layer_kv_cache.GetPtr<int64_t>());
  void* const k_cache_ptr = layer_kv_cache.GetPtr<void*>()[1 + this->layer_index_ * 2];  // block中每层layer的起始地址
  int32_t* const block_table_ptr = block_table.GetPtr<int32_t>();  // block id，加上k_cache_ptr后就是对应的cache block
  const int max_blocks_per_seq = block_table.shape[1];             // shape: [bs, max_num_blocks_per_query]

  InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
      hidden_buffer_1.GetPtr<void>(), output.GetPtr<void>(), q_nope_tensor.GetPtr<void>(), q_pe_tensor.GetPtr<void>(),
      compressed_kv_tensor.GetPtr<void>(), k_pe_tensor.GetPtr<void>(), k_list, kv_seq_len.GetPtr<void>(),
      this->context_->GetComputeStreams()[this->rank_].Get(), cache_offset.GetPtr<void>(), this->num_heads_,
      this->qk_rope_head_dim_, this->kv_lora_rank_, this->block_token_num_, this->k_scale_, this->v_scale_, batch_size,
      rotary_embedding_pos.GetPtr<void>(), rotary_embedding_mask.GetPtr<void>(), total_tokens, this->attn_scale_,
      this->rotary_embedding_cuda_, tile_scheduler_metadata_tensor.GetPtr<void>(), num_splits_tensor.GetPtr<void>(),
      this->rank_, k_cache_ptr, block_table_ptr, kv_cache_block_num, max_blocks_per_seq, q_seq_len);

  // Correctly set the shape for the following bmm
  output.shape = {total_tokens, static_cast<size_t>(num_heads_), static_cast<size_t>(kv_lora_rank_)};
  return Status();
}

}  // namespace ksana_llm
