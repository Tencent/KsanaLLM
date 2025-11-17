/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/attention_kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/singleton.h"

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
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/permute/permute.h"
#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

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
                 std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, int total_tokens,
                 int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size, int stride_size,
                 float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank, int block_size,
                 void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
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
      // reverse rope for flexible cached tokens, with is_reverse flag setting to true.
      // pass 0 params to use default value
      rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(flexible_rotary_embedding_pos_ptr),
                                      reinterpret_cast<int64_t*>(flexible_rotary_embedding_mask_ptr), nullptr,
                                      k_tensor.data_ptr(), total_tokens, stream, 0, 0, 0, /* is_reverse */ true);
      CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
    }

    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask), q_tensor.data_ptr(),
                                    k_tensor.data_ptr(), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
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
    if (enable_blocked_multi_token_forwarding_kv) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets,
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    } else {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    }
  }

  // 统一通过 InvokeMhaVarlenFwd 调用，去掉 flash-attn 版本宏判断
  // 参考 Dao-AILab/flash-attention 的说明：当 out 非空且满足特定条件时会 core dump
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

  std::vector<at::Tensor> mha_output;
  // TODO(qiannanzhou): 需要考虑FA函数是否需要支持 blocked multi-token forwarding，比如FA3，FA2
  if (enable_blocked_multi_token_forwarding_kv) {
    torch::Tensor seqlen_q_tensor = torch::from_blob(without_prefix_offsets, {batch + 1}, int_options);
    auto cache_options = options;
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
                  KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      // cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn);
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

    MhaVarlenFwdParams params;
    params.q = q_tmp_tensor;
    params.k = k_cache_tensor;
    params.v = v_cache_tensor;
    params.out = out_tensor;
    params.seqlen_q = seqlen_q_tensor.to(torch::kInt32);
    params.seqlen_k = seqlen_tensor.to(torch::kInt32);
    params.seqused_k = seqused_k;
    params.block_table = block_table;
    params.alibi_slopes = alibi_slopes_tensor;
    params.max_seqlen_q = max_forwarding_tokens;
    params.max_seqlen_k = max_tokens;
    params.p_dropout = 0.f;
    params.softmax_scale = 1.0 / sqrt(head_size);
    params.zero_tensors = false;
    params.is_causal = is_causal;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.f;
    params.return_softmax = false;
    params.gen = c10::nullopt;
    mha_output = InvokeMhaVarlenFwd(params);
  } else {
    c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq

    // Only use fp8 inference for FA3
    // FA3 is only applicable to Hopper architecture, so it is unnecessary to check the compute capability
    void* k_scale_ptr = nullptr;
    void* v_scale_ptr = nullptr;
    c10::optional<at::Tensor> k_descale = c10::nullopt;
    c10::optional<at::Tensor> v_descale = c10::nullopt;
    // InvokeMhaVarlenFwd will not call FA3 if there is alibi_slopes
    if (IsUsingFA3() && !alibi_slopes.has_value()) {
      if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
        KLLM_THROW("Flash Attention 3 not support fp8_e5m2 KV Cache. Please use fp8_e4m3.");
      } else if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
        if constexpr (!std::is_same<SCALAR_T, __nv_bfloat16>::value) {
          KLLM_THROW("Flash Attention 3 only supports BF16 output for FP8 input.");
        }

        // Quantize Q tensor for E4M3 KV cache type
        KLLM_LOG_DEBUG << "FP8 kv cache and flash attention 3 enabled, using FP8 inference, quantizing q tensor.";

        auto cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat8_e4m3fn);
        const size_t quant_qkv_offset = total_tokens * num_heads * head_size * sizeof(SCALAR_T);
        void* const quant_qkv_ptr = out + quant_qkv_offset;
        torch::Tensor quant_qkv_tensor =
            torch::from_blob(quant_qkv_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, cache_options);
        auto split_quant_qkv_tensors =
            quant_qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
        torch::Tensor quant_q_tensor = split_quant_qkv_tensors[0];
        torch::Tensor quant_k_tensor = split_quant_qkv_tensors[1];
        torch::Tensor quant_v_tensor = split_quant_qkv_tensors[2];

        // Quant q to cache type
        const float q_scale = k_scale;
        llm_kernels::nvidia::ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
            /*q_src*/ reinterpret_cast<SCALAR_T*>(q_tmp_tensor.data_ptr()),
            /*q_dst*/ reinterpret_cast<CACHE_T*>(quant_q_tensor.data_ptr()), total_tokens, num_heads, head_size,
            stride_size, q_scale, stream);

        // Copy kv from cache
        if (use_cache) {
          CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FP8WithPrefixReverseCacheCopy<CACHE_T>(
              reinterpret_cast<CACHE_T*>(quant_k_tensor.data_ptr()),
              reinterpret_cast<CACHE_T*>(quant_v_tensor.data_ptr()), k_list, v_list, reinterpret_cast<size_t*>(seqlen),
              reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size,
              stride_size, sizeof(SCALAR_T), stream));
        } else {
          KLLM_THROW("Please enable kv cache to use flash attention 3 fp8 inference.");
        }

        q_tmp_tensor = torch::reshape(quant_q_tensor, {total_tokens, num_heads, head_size});
        k_tensor = torch::reshape(quant_k_tensor, {total_tokens, num_kv_heads, head_size});
        v_tensor = torch::reshape(quant_v_tensor, {total_tokens, num_kv_heads, head_size});

        // FA3 requires scale inputs in [batch = 1, num_heads]
        // qkv_ptr layout: [qkv]
        // out ptr layout: [out] [quant_qkv] [kv scale]
        if (k_scale != 1.0f || v_scale != 1.0f) {
          KLLM_LOG_DEBUG << "Valid kv scale detected, preparing FA3 scale inputs.";
          const size_t scale_k_offset = total_tokens * (num_heads + num_kv_heads * 2) * head_size * sizeof(CACHE_T);
          const size_t scale_v_offset = num_heads * sizeof(float);
          k_scale_ptr = quant_qkv_ptr + scale_k_offset;
          v_scale_ptr = k_scale_ptr + scale_v_offset;
          llm_kernels::nvidia::InvokeFillKVScaleIntoBuffer(k_scale_ptr, v_scale_ptr, &k_scale, &v_scale, num_kv_heads,
                                                           stream);

          const auto scale_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<float>());
          k_descale = torch::from_blob(k_scale_ptr, {1, num_kv_heads}, scale_options);
          v_descale = torch::from_blob(v_scale_ptr, {1, num_kv_heads}, scale_options);
        }
      }
    }

    MhaVarlenFwdParams params;
    params.q = q_tmp_tensor;
    params.k = torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size});
    params.v = torch::reshape(v_tensor, {total_tokens, num_kv_heads, head_size});
    params.out = out_tensor;
    params.seqlen_q = seqlen_tensor.to(torch::kInt32);
    params.seqlen_k = seqlen_tensor.to(torch::kInt32);
    params.seqused_k = seqused_k;
    params.block_table = block_table;
    params.alibi_slopes = alibi_slopes_tensor;
    params.max_seqlen_q = max_tokens;
    params.max_seqlen_k = max_tokens;
    params.p_dropout = 0.f;
    params.softmax_scale = 1.0 / sqrt(head_size);
    params.zero_tensors = false;
    params.is_causal = is_causal;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.f;
    params.return_softmax = false;
    params.gen = c10::nullopt;
    params.q_descale = k_descale;
    params.k_descale = k_descale;
    params.v_descale = v_descale;
    mha_output = InvokeMhaVarlenFwd(params);
  }

  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(out, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }
}

#define ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                     \
  template void AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                             \
      void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,                \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, int total_tokens,               \
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
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,
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
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask), q_tensor_ptr, k_tensor_ptr,
                                    total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
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
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
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
    // Use unified wrapper InvokeMhaFwdKvcCache instead of version-specific macros
    c10::optional<at::Tensor> seqlen_q_tensor = c10::optional<at::Tensor>(torch::arange(
        0, batch + 1, torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>())));
    c10::optional<at::Tensor> seqlens_k_tensor2 = c10::optional<at::Tensor>(
        torch::from_blob(context_lens_ptr, {batch},
                         torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>())));

    ksana_llm::MhaFwdKVCacheParams fa_params;
    fa_params.q = q_tensor;              // [batch, 1, num_heads, head_size]
    fa_params.k_cache = k_cache_tensor;  // [num_blocks, block_size, num_kv_heads, head_size]
    fa_params.v_cache = v_cache_tensor;
    fa_params.seqlen_q = seqlen_q_tensor;        // cu_seqlens_q: [batch+1]
    fa_params.seqlens_k = seqlens_k_tensor2;     // [batch]
    fa_params.block_table = block_table_tensor;  // [batch, max_blocks_per_seq]
    fa_params.alibi_slopes = alibi_slopes_tensor;
    fa_params.out = out_tensor;  // [batch, 1, num_heads, head_size]
    fa_params.softmax_scale = softmax_scale;
    fa_params.is_causal = true;
    fa_params.window_size_left = -1;
    fa_params.window_size_right = -1;
    fa_params.softcap = 0.0f;

    InvokeMhaFwdKvcCache(fa_params);

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
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, void* workspace_ptr,           \
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
