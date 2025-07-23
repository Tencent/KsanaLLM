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
#include "csrc/kernels/nvidia/marlin_moe/fused_marlin_moe.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "csrc/kernels/nvidia/weight_only_batched_gemv/weight_only_gemv_wrapper.h"
#include "csrc/utils/nvidia/scalar_type.hpp"
#include "csrc/utils/quant_type.h"

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

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

template <typename T>
void InvokeGroupedTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, int num_experts,
                       int topk, bool renormalize, int num_expert_group, int topk_group, std::string scoring_func,
                       void* e_bias, float routed_scaling_factor, int rank, cudaStream_t stream);

template <typename T, bool UseExpertParallel>
void InvokeFusedMoe(void* hidden_states, void* w1, void* w2, int* expert_map, int topk,
                    bool renormalize, const std::string& scoring_func_, void* e_bias, bool inplace,
                    bool use_grouped_topk, int num_expert_group, int topk_group, DataType weight_dtype,
                    DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, void* w2_scale,
                    void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,
                    std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr,
                    float routed_scaling_factor, void* output_hidden_states, void* intermediate_cache1,
                    void* intermediate_cache2, void* intermediate_cache3, void* fused_id_buffer, int num_tokens,
                    int num_experts, int hidden_size, int inter_size, size_t world_expert_para_size,
                    void* dequant_workspace, int rank, cudaStream_t stream);

size_t InvokeGetFusedMarlinMoeWorkspaceSize(int num_tokens, int inter_size, int hidden_size, int num_experts, int topk,
                                            size_t data_type_size);

void FusedMarlinMoe(half* output, const half* input, const float* gating_output, const void* w1, const void* w2,
                    const void* w1_scale, const void* w2_scale, void* workspace, size_t workspace_size, int num_tokens,
                    int inter_size, int hidden_size, int num_experts, int topk, cudaStream_t& stream,
                    llm_kernels::nvidia::MOEExpertScaleNormalizationMode norm_mode, const int* g_idx1,
                    const int* g_idx2, const void* sort_indices1, const void* sort_indices2, const void* w1_zeros,
                    const void* w2_zeros, int num_bits, int group_size);

}  // namespace ksana_llm
