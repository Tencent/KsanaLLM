/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/attention_kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/utils/singleton.h"
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
#  include "flash_api.h"
#endif

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"
#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/fused_moe_gptq_int4_fp8_kernel/dequant.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/moe/moe.h"
#include "csrc/kernels/nvidia/moe_wna16/moe_wna16.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/per_token_group_quant/per_token_group_quant_8bit.h"
#include "csrc/kernels/nvidia/permute/permute.h"
#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/search_status.h"

namespace ksana_llm {

// Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
// intended for use in testing accuracy outcomes only.
bool kContextDecodeUseFP8Cache = []() -> bool {
  const char* val = std::getenv("ContextDecodeUseFP8Cache");
  if (val != nullptr) {
    return true;
  }
  return false;
}();

template <typename T>
void InvokeQKRmsNorm(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps,
                     const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                     const int32_t head_size, const int64_t* mask, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFusedQKVRmsNorm<T>(
      reinterpret_cast<T*>(qkv_ptr), reinterpret_cast<const T*>(qkv_ptr), reinterpret_cast<const T*>(q_gamma),
      reinterpret_cast<const T*>(k_gamma), layernorm_eps, total_tokens, num_heads, num_kv_heads, head_size, mask,
      stream));
}
#define INVOKE_QK_LAYER_NORM(T)                                                                                        \
  template void InvokeQKRmsNorm<T>(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps, \
                                   const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,    \
                                   const int32_t head_size, const int64_t* mask, cudaStream_t stream)
INVOKE_QK_LAYER_NORM(float);
INVOKE_QK_LAYER_NORM(half);
INVOKE_QK_LAYER_NORM(__nv_bfloat16);
#undef INVOKE_QK_LAYER_NORM

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,
                 std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                 int total_tokens, int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size,
                 int stride_size, float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank,
                 int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                 const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                 void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                 void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                 void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                 void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                 void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                 size_t* without_prefix_offsets, int max_forwarding_tokens, bool enable_qk_pre_norm_before_rotary_pos,
                 bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale,
                 bool enable_blocked_multi_token_forwarding_kv) {
  // qk norm before rotary position embedding
  if (enable_qk_pre_norm_before_rotary_pos && q_norm_weight != nullptr && k_norm_weight != nullptr) {
    InvokeQKRmsNorm<SCALAR_T>(qkv_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  torch::Tensor qkv_tensor =
      torch::from_blob(qkv_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  c10::optional<at::Tensor> null_tensor = c10::nullopt;
  c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;

  // rotary embedding
  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  if (flexible_len != 0) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FlexibleReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<CACHE_T**>(src_flexible_kv_cache_ptr), reinterpret_cast<CACHE_T**>(dst_flexible_kv_cache_ptr),
        reinterpret_cast<int*>(src_flexible_token_idx_ptr), reinterpret_cast<int*>(dst_flexible_token_idx_ptr),
        block_size, layer_index, flexible_len, num_kv_heads, head_size, stride_size, stream));
  }

  if (use_cache && !enable_blocked_multi_token_forwarding_kv) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
        v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets),
        reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
        k_scale, v_scale, stream));
  }

  if (!no_rope && rotary_embedding_cuda.has_value()) {
    if (flexible_len != 0) {
      rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(flexible_rotary_embedding_pos_ptr),
                                      reinterpret_cast<int64_t*>(flexible_rotary_embedding_mask_ptr), nullptr,
                                      reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), total_tokens, stream);
      CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
    }

    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                    reinterpret_cast<SCALAR_T*>(q_tensor.data_ptr()),
                                    reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }

  if (!no_rope && use_qk_norm) {
    InvokeQKRmsNorm<SCALAR_T>(qkv_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }

  if (attn_temperature_tuning) {
    torch::Tensor positions_tensor =
        torch::from_blob(rotary_embedding_pos, {total_tokens}, int_options).to(torch::kFloat32);
    torch::Tensor attn_scale_tensor =
        torch::log(torch::floor((positions_tensor + 1.0f) / static_cast<float>(floor_scale)) + 1.0f) * attn_scale +
        1.0f;
    attn_scale_tensor = attn_scale_tensor.unsqueeze(-1).to(q_tensor.dtype());
    // Notice: attn_scale_tensor and q_tensor's multiplication is fp32 in transformers
    torch::mul_out(q_tensor, q_tensor, attn_scale_tensor);
  }

  if (use_cache) {
    if (enable_blocked_multi_token_forwarding_kv)
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets,
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    else
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
  }

// flash attention 2 or flash attention 1
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)
  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  }
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }
  // vllm-attn apis.
#  ifdef ENABLE_VLLM_FLASH_ATTN_MINOR_6
  std::vector<at::Tensor> mha_output;
  if (enable_blocked_multi_token_forwarding_kv) {
    torch::Tensor seqlen_q_tensor = torch::from_blob(without_prefix_offsets, {batch + 1}, int_options);
    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      // cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt8);
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
    mha_output = mha_varlen_fwd(q_tmp_tensor, k_cache_tensor, v_cache_tensor, out_tensor,
                                seqlen_q_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k,
                                block_table, alibi_slopes_tensor, max_forwarding_tokens, max_tokens, 0.f,
                                1.0 / sqrt(head_size), false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  } else {
    c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq
    mha_output = mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                                torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                                seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k,
                                block_table, alibi_slopes_tensor, max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size),
                                false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  }
#  endif

  // flash_attn v.2.4, 2.5.6. and later versions.
#  if defined(ENABLE_FLASH_ATTN_2)
#    if defined(ENABLE_FLASH_ATTN_MINNOR_4) || defined(ENABLE_FLASH_ATTN_MINOR_5)
  std::vector<at::Tensor> mha_output =
      mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                     torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                     seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k, alibi_slopes_tensor,
                     max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);

#    else  // Since v2.7.2.post1, add two more parms, such as block_table_,leftpad_t_.
  std::vector<at::Tensor> mha_output = mha_varlen_fwd(
      q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
      torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor, seqlen_tensor.to(torch::kInt32),
      seqlen_tensor.to(torch::kInt32), seqused_k, const_null_tensor, /* leftpad_k_  */
      null_tensor,                                                   /* block_table */
      alibi_slopes_tensor, max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, 0.0, false,
      c10::nullopt);
#    endif
#  endif
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(out, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

#else  // flash_attn v1.x?
  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  flash_attn::mha_varlen_fwd(torch::reshape(q_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                             torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                             seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), max_tokens, max_tokens,
                             0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
#endif
}

#define ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                     \
  template void AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                             \
      void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,                \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, int total_tokens,     \
      int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size, int stride_size, float k_scale,      \
      float v_scale, size_t tensor_para_size, bool is_causal, int rank, int block_size, void** k_list, void** v_list, \
      void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes, int layer_index,           \
      void* flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,                              \
      void* dst_flexible_kv_cache_ptr, void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr,             \
      void* src_flexible_token_idx_ptr, void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps,      \
      bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream,                \
      void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,                     \
      int max_blocks_per_seq, size_t* without_prefix_offsets, int max_forwarding_tokens,                              \
      bool enable_qk_pre_norm_before_rotary_pos, bool no_rope, bool attn_temperature_tuning, float attn_scale,        \
      size_t floor_scale, bool enable_blocked_multi_token_forwarding_kv)
ATTEN_VARLEN(float, float, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
ATTEN_VARLEN(half, half, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
ATTEN_VARLEN(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef ATTEN_VARLEN

#define PAGED_ATTENTION(T1, T2, CACHE_T1, CACHE_T2, KV_DTYPE)                                                        \
  template <>                                                                                                        \
  void PagedAttentionOp<T1, CACHE_T1, KV_DTYPE>(                                                                     \
      int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale, \
      void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,          \
      void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream, void* workspace,              \
      size_t work_size, const float* alibi_slopes_ptr) {                                                             \
    llm_kernels::nvidia::PagedAttentionCuda<T2, CACHE_T2, KV_DTYPE> op;                                              \
    op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size, k_scale, v_scale);                     \
    op.SetInput(reinterpret_cast<T2*>(out), reinterpret_cast<const T2*>(q_tensor_ptr),                               \
                reinterpret_cast<CACHE_T2**>(key_cache_ptrs), reinterpret_cast<CACHE_T2**>(value_cache_ptrs),        \
                reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),     \
                max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);                          \
    CUDA_CHECK_LAST_ERROR(op.Forward());                                                                             \
  }
PAGED_ATTENTION(float, float, float, float, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
PAGED_ATTENTION(half, uint16_t, half, uint16_t, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#undef PAGED_ATTENTION

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs,
                          void* context_lens_ptr, int max_context_len, cudaStream_t stream, void* cache_offsets_ptr,
                          int seqs_num, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                          float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
                          void* rotary_embedding_mask, int total_tokens,
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                          void* workspace_ptr, float layernorm_eps, bool use_qk_norm, void* q_norm_weight,
                          void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr,
                          int64_t kv_cache_block_num, int max_blocks_per_seq, bool enable_qk_pre_norm_before_rotary_pos,
                          bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale,
                          bool enable_blocked_multi_token_forwarding_kv) {
  // qk norm before rotary position embedding for paged attention
  if (enable_qk_pre_norm_before_rotary_pos && q_norm_weight != nullptr && k_norm_weight != nullptr) {
    InvokeQKRmsNorm<SCALAR_T>(query_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor qkv_tensor =
      torch::from_blob(query_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);

  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  void* q_tensor_ptr = q_tensor.data_ptr();
  void* k_tensor_ptr = k_tensor.data_ptr();
  void* v_tensor_ptr = v_tensor.data_ptr();

  if (!no_rope && rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }
  if (!no_rope && use_qk_norm) {
    InvokeQKRmsNorm<SCALAR_T>(query_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }

  if (attn_temperature_tuning) {
    torch::Tensor positions_tensor =
        torch::from_blob(rotary_embedding_pos, {total_tokens}, int_options).to(torch::kFloat32);
    torch::Tensor attn_scale_tensor =
        torch::log(torch::floor((positions_tensor + 1.0f) / static_cast<float>(floor_scale)) + 1.0f) * attn_scale +
        1.0f;
    attn_scale_tensor = attn_scale_tensor.unsqueeze(-1).to(q_tensor.dtype());
    // Notice: attn_scale_tensor and q_tensor's multiplication is fp32 in transformers
    torch::mul_out(q_tensor, q_tensor, attn_scale_tensor);
  }

  if (enable_blocked_multi_token_forwarding_kv) {
    constexpr size_t kReqQLen = 1;
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
        value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
        block_size, batch, kReqQLen, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
  } else {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
        value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
        block_size, batch, total_tokens, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
  }

  if (enable_blocked_multi_token_forwarding_kv) {
    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, 2, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table_tensor =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
    c10::optional<const at::Tensor> seqlens_k_tensor =
        c10::optional<const at::Tensor>(torch::from_blob(context_lens_ptr, {batch}, int32_options));
    q_tensor = q_tensor.reshape({batch, 1, num_heads, head_size});
    c10::optional<at::Tensor> out_tensor = torch::from_blob(output_ptr, {batch, 1, num_heads, head_size}, options);
    float softmax_scale = 1.0 / sqrt(head_size);
    c10::optional<at::Tensor> null_tensor = c10::nullopt;
    c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
    if (alibi_slopes.has_value()) {
      alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
    }

    //  Not support flash-attn < 2.6.0 && != 2.5.6
#if defined(ENABLE_FLASH_ATTN_2) && defined(ENABLE_FLASH_ATTN_MINOR_5)
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, true, 0);
#elif defined(ENABLE_FLASH_ATTN_2) && defined(ENABLE_FLASH_ATTN_MINOR_7)
    // add leftpad_k_ param since flash-attn 2.7.2
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    const_null_tensor,    // indices that the KV cache starts. [batch_size,], nullptr, default 0
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, 0.0, true, 0);

#else  // It works for vllm-flash-attn and flash-attn >= v2.5.6
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, 0.0, true, 0);
#endif

  } else {
    const float* alibi_slopes_ptr =
        reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
    PagedAttentionOp<SCALAR_T, CACHE_T, KV_DTYPE>(num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale,
                                                  v_scale, output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
                                                  cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num,
                                                  stream, workspace_ptr, work_size, alibi_slopes_ptr);
  }
}

#define RUN_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                             \
  template void InvokePagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                                   \
      void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,     \
      int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads, int head_size, \
      int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale, int batch,                    \
      void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens,                                     \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr, \
      float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,   \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,           \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,                                  \
      bool enable_qk_pre_norm_before_rotary_pos, bool no_rope, bool attn_temperature_tuning, float attn_scale,       \
      size_t floor_scale, bool enable_blocked_multi_token_forwarding_kv)
RUN_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#undef RUN_PAGED_ATTENTION

}  // namespace ksana_llm
