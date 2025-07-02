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
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaAttenVarlen(void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,
                    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight,
                    void* kv_b_nope_weight_scale, void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim,
                    void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
                    void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen, float attn_scale,
                    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                    int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim,
                    int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int num_kv_heads, int head_size,
                    int stride_size, float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank,
                    int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                    const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                    void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                    void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                    void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                    void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                    void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                    size_t* without_prefix_offsets, int max_forwarding_tokens, int total_prefix_len,
                    void* seqlens_q_ptr, void* prefix_k_buffer, void* prefix_v_buffer, void* prefix_o_buffer,
                    void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer,
                    QuantMode mm_quant_mode);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeMlaPagedAttention(
    void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight, void* kv_b_nope_weight_scale,
    void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim, void* workspace, cublasHandle_t& cublas_handles,
    cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,
    int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads,
    int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int head_size, int num_kv_heads,
    int stride_size, int block_size, float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
    void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr,
    float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,
    const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,
    int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, QuantMode mm_quant_mode,
    int q_seq_len);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeAbsorbMlaPagedAttention(
    void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* w_q_uk_weight, void* w_q_r_weight, void* w_uv_weight, void* w_o_weight, void* o_weight_scale,
    void* w_uv_o_weight, void* w_q_uk_weight_scale, void* w_q_r_weight_scale, void* w_uv_o_weight_scale,
    size_t w_uv_o_dim, void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
    void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr, cudaStream_t stream,
    void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,
    int v_head_dim, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale,
    int batch, void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
    void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace_ptr, float layernorm_eps, bool use_qk_norm,
    void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
    void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,
    int max_blocks_per_seq, int q_seq_len, int skiped_decode_q_token, int total_q_len, QuantMode mm_quant_mode);

template <typename T>
void MlaAbsorbWeight(void* w_q, void* w_uk, void* w_uv, void* w_o, void* w_q_uk, void* w_uv_o, size_t q, size_t n,
                     size_t d, size_t l, size_t h, bool transpose_matrix, int rank, cudaStream_t& stream);

void SetMlaMetadataKernelAttribute(const int max_batch_size, cudaStream_t stream);

}  // namespace ksana_llm
