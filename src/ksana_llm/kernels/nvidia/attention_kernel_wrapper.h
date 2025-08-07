/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <optional>
#include <vector>

#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"
#include "csrc/kernels/nvidia/machete/machete_wrapper.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "csrc/kernels/nvidia/weight_only_batched_gemv/weight_only_gemv_wrapper.h"
#include "csrc/utils/nvidia/scalar_type.hpp"
#include "csrc/utils/quant_type.h"

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

template <typename T>
void InvokeQKRmsNorm(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps,
                     const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                     const int32_t head_size, const int64_t* mask, cudaStream_t stream);

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
                 bool enable_blocked_multi_token_forwarding_kv);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttentionOp(int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale,
                      float v_scale, void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs,
                      void* cache_offsets_ptr, void* context_lens_ptr, int max_context_len, int num_seqs,
                      cudaStream_t& stream, void* workspace, size_t work_size, const float* alibi_slopes_ptr);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(void* out,                // [num_seqs, num_heads, head_size]
                          void* query,              // [num_seqs, num_heads, head_size]
                          void** key_cache_ptrs,    // num_seqs,[seq_blocks]
                          void** value_cache_ptrs,  // num_seqs,[seq_blocks]
                          void* context_lens_ptr,   // [num_seqs]
                          int max_context_len, cudaStream_t stream,
                          void* cache_offsets_ptr,  // num_seqs
                          int num_seqs, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                          float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
                          void* rotary_embedding_mask, int total_tokens,
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,
                          void* workspace, float layernorm_eps, bool use_qk_norm, void* q_norm_weight,
                          void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr,
                          int64_t kv_cache_block_num, int max_blocks_per_seq, bool enable_qk_pre_norm_before_rotary_pos,
                          bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale,
                          bool enable_blocked_multi_token_forwarding_kv);

}  // namespace ksana_llm
