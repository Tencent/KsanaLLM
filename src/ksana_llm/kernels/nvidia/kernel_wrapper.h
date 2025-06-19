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

#define BITS_PER_BYTE 8

void DequantInt4Fp8(cudaStream_t stream, void* output, const void* input, const size_t datasize);

void UpdateMoeWna16BlockConfig(std::unordered_map<std::string, int>& config, bool use_moe_wna16_cuda,
                               bool use_int4_w4a8, int num_valid_tokens, int size_k, int size_n, int num_experts,
                               int group_size, int real_top_k, int block_size_m);

bool ShouldMoeWna16UseCuda(int num_valid_tokens, int group_size, int num_experts, int bit);

template <typename T>
void InvokeMoeWna16Gemm(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                        const void* b_scales, const void* b_qzeros, const void* topk_weights,
                        const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_pad,
                        int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts,
                        int size_m, int size_n, int size_k, int group_size, int num_token_blocks);

template <typename T>
llm_kernels::nvidia::vllm_dtype::ScalarType GetMacheteDataType();

std::vector<std::string> GetMacheteSupportedSchedules(
    llm_kernels::nvidia::vllm_dtype::ScalarType a_type, llm_kernels::nvidia::vllm_dtype::ScalarType b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_scales_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_zeros_type);

void InvokeMacheteGemm(int64_t& workspace_size, void* workspace, cudaStream_t stream, int M, int N, int K,
                       const void* Aptr, const void* Bptr, void* Dptr,
                       llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
                       llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type,
                       std::optional<void*> const& maybe_group_scales_ptr,
                       std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                       std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type,
                       std::optional<void*> const& maybe_group_zeros_ptr,
                       std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                       std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                       std::optional<int64_t> maybe_group_size, std::optional<std::string> maybe_schedule);

void InvokeMachetePrepackWeight(
    const void* B_ptr, const std::vector<size_t>& B_shape, void* out_ptr,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type, cudaStream_t stream);

std::string GetMacheteBestSchedule(
    size_t warmup_iters, size_t record_iters, void* workspace, cudaStream_t stream, int M, int N, int K,
    const void* Aptr, const void* Bptr, void* Dptr, llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type, std::optional<void*> const& maybe_group_scales_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type,
    std::optional<void*> const& maybe_group_zeros_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type,
    std::optional<int64_t> maybe_group_size);

void InvokeMarlinAwqRepack(const void* b_q_weight_ptr, void* out_ptr, int64_t size_k, int64_t size_n, int64_t num_bits,
                           int rank, cudaStream_t stream);

std::vector<int64_t> GetMarlinAwqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits);

void InvokeMarlinGptqRepack(const void* b_q_weight_ptr, const void* perm_ptr, void* out_ptr, int64_t size_k,
                            int64_t size_n, int64_t num_bits, bool has_perm, int rank, cudaStream_t stream);

std::vector<int64_t> GetMarlinGptqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits);

template <typename T>
llm_kernels::nvidia::marlin::WorkspaceInfo GetMarlinWorkspace(bool use_fp32_reduce, bool has_act_order, int rank,
                                                              int64_t size_m, int64_t size_k);

template <typename T>
void InvokeMarlinPermuteScales(cudaStream_t stream, const void* input, void* output, const size_t k, const size_t n,
                               const int64_t groupsize);

template <typename T>
void InvokeMarlinGemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
                      bool has_zp, bool has_act_order, bool is_awq, int rank, cudaStream_t stream);

template <typename T>
torch::ScalarType GetTorchDataType();

DataType GetDataTypeFromTorchType(const c10::ScalarType& torch_type);

template <typename T, typename WT, typename OT>
void GetMoeGemmWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size, size_t expert_inter_size,
                             size_t expert_topk, int tp_size, int rank, bool use_lora, size_t& ws_bytes);

template <typename T, typename WT, typename OT>
size_t InvokeMoeGemmConfigProfile(bool is_fp8 = false);

template <typename T, typename WT, typename OT, llm_kernels::nvidia::MOEExpertScaleNormalizationMode NT>
void InvokeMoeCutlassGemm(void const* input_activations, void* gating_output, void const* fc1_expert_weights,
                          void const* fc2_expert_weights, void* e_score_correction_bias, int64_t const num_rows,
                          int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const topk,
                          char* workspace_ptr, void* final_output, void* token_topk_final_scales,
                          int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row, int tp_size,
                          int rank, bool use_lora, size_t best_config_index, bool use_vllm_moe_,
                          uint32_t num_expert_group_, uint32_t expert_groups_topk_, const std::string& scoring_func_,
                          const std::string& topk_method_, bool norm_topk_prob_, float routed_scaling_factor_,
                          bool use_e_score_correction_bias_, cudaStream_t stream, bool is_fp8 = false,
                          void const* scale1 = nullptr, void const* scale2 = nullptr, void const* scale3 = nullptr,
                          bool apply_weight = false);

template <typename T, llm_kernels::nvidia::WeightType WT>
void GetFpAIntBGroupCutlassGemmWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes);

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCutlassGemm(void* output, const void* input, const void* weight, const void* scales,
                                   const void* zeros, void* ws, size_t m, size_t n, size_t k, size_t groupsize,
                                   size_t config_index, cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
size_t InvokeFpAIntBGroupCutlassGemmConfigProfile(size_t warmup, size_t iter, void* output, const void* input,
                                                  const void* weight, const void* scales, const void* zeros, void* ws,
                                                  size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
bool GetFpAIntBGroupCudaGemmSupported();

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCudaGemm(void* output, const void* input, const void* weight, const void* scales,
                                const void* zeros, size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream);

// Invoke the lookup embedding.
template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output, const T emb_scale, int vocab_size,
                     int hidden_size, int bs, int vocab_id, cudaStream_t stream, void* workspace_ptr = nullptr);

// Layernorm without bias computes rmsnorm.
template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const void* bias, const float layernorm_eps, const int m,
                     const int n, void* output, cudaStream_t stream);

template <typename T>
void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream, void* workspace_ptr,
                  cublasLtMatmulAlgo_t* cublaslt_algo);

template <typename T>
void InvokeBatchedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int batch_size, int m, int n,
                         int k, const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream,
                         void* workspace_ptr, size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo);

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream);

// for the tensor input with shape [m, n]
//  out = act(input[:, :n/2]) * input[:, n/2:]
template <template <typename T> class Activation, typename T>
void InvokeRowBasedGatedActivation(const void* input, const int m, const int n, void* output, cudaStream_t stream);

// Invoke activation in-place, `output` must be the same as `input`.
template <template <typename T> class Activation, typename T>
void InvokeGatedActivation(const void* input, const void* bias, const void* gated_weights, const void* gated_bias,
                           const int m, const int n, void* output, cudaStream_t stream);

template <typename T>
void AssembleTokensHidden(const void* inputs, const void* logits_idx, const int batch_size, const int hidden_units_num,
                          void* output, cudaStream_t& stream);

template <typename T>
void Concat(const void* input_a, const void* input_b, size_t concat_size_a, size_t concat_size_b, size_t outer_dim_size,
            size_t inner_dim_size, void* output, cudaStream_t& stream);

template <typename T>
void InvokeQKRmsNorm(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps,
                     const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                     const int32_t head_size, const int64_t* mask, cudaStream_t stream);

template <typename T>
void Expand(void* input, void* output, const int m, const int expand_size, const int n, const size_t stride,
            cudaStream_t stream);

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
                 void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream,
                 void* k_cache_ptr = nullptr, void* v_cache_ptr = nullptr, int32_t* block_table_ptr = nullptr,
                 int64_t kv_cache_block_num = 0, int max_blocks_per_seq = 0, size_t* without_prefix_offsets = nullptr,
                 int max_forwarding_tokens = 0, bool enable_qk_pre_norm_before_rotary_pos = false, bool no_rope = false,
                 bool attn_temperature_tuning = false, float attn_scale = 0, size_t floor_scale = 0);

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
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                          void* workspace, float layernorm_eps, bool use_qk_norm, void* q_norm_weight,
                          void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace, void* k_cache_ptr = nullptr, void* v_cache_ptr = nullptr,
                          int32_t* block_table_ptr = nullptr, int64_t kv_cache_block_num = 0,
                          int max_blocks_per_seq = 0, bool enable_qk_pre_norm_before_rotary_pos = false,
                          bool no_rope = false, bool attn_temperature_tuning = false, float attn_scale = 0,
                          size_t floor_scale = 0);

template <typename T>
void CustomAllReduceInit(void** ptr, void** signals, void* rank_data, size_t rank_data_sz, int cur_rank,
                         int total_ranks, bool is_full_nvlink, uint32_t root_rank);

template <typename T>
void CustomAllReduceRegisterBuffer(void* ptr, void** input_handles, cudaStream_t& stream);

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);

template <typename T>
void InvokeSigmoidActivation(void* input, const size_t size, const float scale, cudaStream_t& stream);

template <typename T>
ncclDataType_t GetNcclDataType();

template <typename T>
void DataToFloat(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                 void* output, cudaStream_t& stream);

template <typename T>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream);

template <typename T>
void InvokeMul(void* a, void* b, void* c, int m1, int n1, int m2, int n2, int device_rank);
// c = InvokeMul(a, b)
void InvokeMul(float* a, float* b, float* c, int n, int device_rank);
void Reciprocal(float* out, float* in, int n, int device_rank);
void Max(float* out, float* a, float* b, int n, int device_rank);

void CalcLogprobs(float* logits, float* temperatures, int vocab_size, int bs, int logprobs_num, float* logprobs,
                  int64_t* token_ids);

#ifdef ENABLE_FP8
template <typename T>
void Fp8E4m3Quantize(int num_channels, int channel_size, const T* input_ptr, void* quant_ptr, float* scale_ptr,
                     bool is_static, cudaStream_t& stream);

template <typename T>
void Fp8QuantizedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                        const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale, T* c_ptr,
                        cudaStream_t& stream, cublasLtMatmulAlgo_t* cublaslt_algo, void* workspace);

void RescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                    cudaStream_t& stream);
template <typename T>
void ScaledQuantizeFp8E4m3(T* x, void* output, float* scale, std::vector<size_t> group_shape, int m, int n, int rank);

template <typename T>
void DequantFp8E4m3BlockWise(const void* d_data, const void* d_s, void* d_output, int m, int n, int block_size,
                             cudaStream_t& stream);
#endif

size_t InvokeGetCublasWorkspaceSize();

#ifdef ENABLE_VLLM_FLASH_ATTN_2
cudaStream_t InvokeSetTorchStream(cudaStream_t& stream, int rank);
#endif

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

size_t GetWorkspaceBuffer(int tokens_num, int num_heads, int seqlen_q, int head_size_v);

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
void InvokeBlockGemm(void* a, float* a_scales, void* b, float* b_scales, void* output, int m, int k, int n,
                     cudaStream_t& stream, void* cutlass_buffer = nullptr, size_t cutlass_buffer_size = 0ul);

template <typename T>
void MlaAbsorbWeight(void* w_q, void* w_uk, void* w_uv, void* w_o, void* w_q_uk, void* w_uv_o, size_t q, size_t n,
                     size_t d, size_t l, size_t h, bool transpose_matrix, int rank, cudaStream_t& stream);

template <typename T>
void InvokeGroupedTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, int num_experts,
                       int topk, bool renormalize, int num_expert_group, int topk_group, std::string scoring_func,
                       void* e_bias, float routed_scaling_factor, int rank, cudaStream_t stream);

template <typename T, bool UseExpertParallel>
void InvokeFusedMoe(void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk,
                    bool renormalize, const std::string& scoring_func_, void* e_bias, bool inplace,
                    bool use_grouped_topk, int num_expert_group, int topk_group, DataType weight_dtype,
                    DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, void* w2_scale,
                    void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,
                    std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr,
                    float routed_scaling_factor, void* output_hidden_states, void* intermediate_cache1,
                    void* intermediate_cache2, void* intermediate_cache3, void* fused_id_buffer, int num_tokens,
                    int num_experts, int hidden_size, int inter_size, void* dequant_workspace, int rank,
                    cudaStream_t stream);

template <typename T>
void InvokePerTokenGroupQuantFp8E4m3(void* input, void* output_q, void* output_s, int m, int n, bool is_column_major,
                                     cudaStream_t stream, int64_t group_size = 128, float eps = 1e-10,
                                     float min_fp8 = -448.0, float max_fp8 = 448.0);

template <typename T>
void InvokeFusedAddRmsNorm(void* input, void* residual, void* weight, double eps, int m, int n, cudaStream_t stream);

}  // namespace ksana_llm
