/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/moe_kernel_wrapper.h"

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

// Adapted from https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/fused_moe/fused_moe.py#L676
void UpdateMoeWna16BlockConfig(std::unordered_map<std::string, int>& config, bool use_moe_wna16_cuda,
                               bool use_fp8_compute, int num_valid_tokens, int size_k, int size_n, int num_experts,
                               int group_size, int real_top_k, int block_size_m) {
  if (config.find("block_size_n") != config.end() && config.find("block_size_k") != config.end()) {
    // optimal block config is set
    return;
  }

  if (!use_moe_wna16_cuda) {
    if (use_fp8_compute) {
      config["block_size_n"] = 128;
      config["block_size_k"] = 128;
    } else {
      // triton moe wna16 kernel
      if (num_valid_tokens / real_top_k == 1) {
        // if bs=1, use a smaller block_size_n
        config["block_size_n"] = 32;
        config["block_size_k"] = 64;
      } else {
        config["block_size_n"] = 64;
        config["block_size_k"] = 32;
      }
    }
  } else {
    // cuda moe wna16 kernel
    // set default block_size 128, and increase them when num_blocks is too large.
    int block_size_n = 128;
    int block_size_k = 128;
    if (block_size_k <= group_size) {
      block_size_k = group_size;
    }

    int num_n_blocks = size_k / block_size_k;
    int num_k_blocks = size_n / block_size_k;
    int num_m_blocks = (num_valid_tokens + block_size_m - 1) / block_size_m + num_experts;
    if (num_valid_tokens / real_top_k <= block_size_m) {
      num_m_blocks = std::min(num_m_blocks, num_valid_tokens);
    }
    int num_blocks = num_m_blocks * num_n_blocks * num_k_blocks;

    if (size_k % 256 == 0 && num_blocks >= 256 && block_size_k < 256) {
      block_size_k = 256;
      num_blocks = num_blocks / (256 / block_size_k);
    }

    if (num_m_blocks <= 16 && size_k % (block_size_k * 2) == 0 && block_size_k <= 512 && num_blocks >= 512) {
      block_size_k *= 2;
      num_blocks /= 2;
    }

    if (num_blocks > 1024) {
      block_size_n = 256;
      num_n_blocks /= 2;
      num_blocks /= 2;
    }

    if (size_n <= 1024 && num_blocks >= 1024) {
      // The kernel performance got much better with block_size_n=1024
      // when num_blocks is large, even when N is small.
      block_size_n = 1024;
    }

    config["block_size_n"] = block_size_n;
    config["block_size_k"] = block_size_k;
  }
}

// Adapted from https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/fused_moe/fused_moe.py#L734
bool ShouldMoeWna16UseCuda(int num_valid_tokens, int group_size, int num_experts, int bit) {
  return bit == 4 && (group_size == 32 || group_size == 64 || group_size == 128) &&
         static_cast<double>(num_valid_tokens) / num_experts <= 6.0;
}

template <typename T>
void InvokeMoeWna16Gemm(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                        const void* b_scales, const void* b_qzeros, const void* topk_weights,
                        const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_pad,
                        int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts,
                        int size_m, int size_n, int size_k, int group_size, int num_token_blocks) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::moe_wna16::moe_wna16_gemm<T>(
      stream, output, input, b_qweight, b_scales, b_qzeros, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_pad, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, bit, num_experts, size_m, size_n, size_k,
      group_size, num_token_blocks));
}
#define INVOKE_MOE_WNA16_GEMM(T)                                                                                 \
  template void InvokeMoeWna16Gemm<T>(                                                                           \
      cudaStream_t stream, void* output, const void* input, const void* b_qweight, const void* b_scales,         \
      const void* b_qzeros, const void* topk_weights, const void* sorted_token_ids, const void* expert_ids,      \
      const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, \
      int num_experts, int size_m, int size_n, int size_k, int group_size, int num_token_blocks)
INVOKE_MOE_WNA16_GEMM(float);
INVOKE_MOE_WNA16_GEMM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_MOE_WNA16_GEMM(__nv_bfloat16);
#endif
#undef INVOKE_MOE_WNA16_GEMM

template <typename T, typename WT, typename OT>
void GetMoeGemmWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size, size_t expert_inter_size,
                             size_t expert_topk, int tp_size, int rank, bool use_lora, size_t& ws_bytes) {
  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  moe_gemm.GetWorkspaceSize(token_num, expert_num, expert_hidden_size, expert_inter_size, expert_topk, tp_size, rank,
                            use_lora, ws_bytes);
}
#define GET_MOE_GEMM_WORKSPACE_SIZE(T, WT, OT)                                                                     \
  template void GetMoeGemmWorkspaceSize<T, WT, OT>(size_t token_num, size_t expert_num, size_t expert_hidden_size, \
                                                   size_t expert_inter_size, size_t expert_topk, int tp_size,      \
                                                   int rank, bool use_lora, size_t& ws_bytes)
GET_MOE_GEMM_WORKSPACE_SIZE(float, float, float);
GET_MOE_GEMM_WORKSPACE_SIZE(half, half, half);
#ifdef ENABLE_FP8
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_fp8_e4m3, __nv_fp8_e4m3, half);
#endif
#ifdef ENABLE_BFLOAT16
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#  ifdef ENABLE_FP8
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16);
#  endif
#endif
#undef GET_MOE_GEMM_WORKSPACE_SIZE

template <typename T, typename WT, typename OT>
size_t InvokeMoeGemmConfigProfile(bool is_fp8) {
  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  return moe_gemm.GetBestConfigIndex(is_fp8);
}
#define INVOKE_MOE_GEMM_CONFIG_PROFILE(T, WT, OT) template size_t InvokeMoeGemmConfigProfile<T, WT, OT>(bool is_fp8)
INVOKE_MOE_GEMM_CONFIG_PROFILE(float, float, float);
INVOKE_MOE_GEMM_CONFIG_PROFILE(half, half, half);
#ifdef ENABLE_FP8
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_fp8_e4m3, __nv_fp8_e4m3, half);
#endif
#ifdef ENABLE_BFLOAT16
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#  ifdef ENABLE_FP8
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16);
#  endif
#endif
#undef INVOKE_MOE_GEMM_CONFIG_PROFILE

template <typename T, typename WT, typename OT, llm_kernels::nvidia::MOEExpertScaleNormalizationMode NT>
void InvokeMoeCutlassGemm(void const* input_activations, void* gating_output, void const* fc1_expert_weights,
                          void const* fc2_expert_weights, void* e_score_correction_bias, int64_t const num_rows,
                          int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const topk,
                          char* workspace_ptr, void* final_output, void* token_topk_final_scales,
                          int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row, int tp_size,
                          int rank, bool use_lora, size_t best_config_index, bool use_vllm_moe_,
                          uint32_t num_expert_group_, uint32_t expert_groups_topk_, const std::string& scoring_func_,
                          const std::string& topk_method_, bool norm_topk_prob_, float routed_scaling_factor_,
                          bool use_e_score_correction_bias_, cudaStream_t stream, bool is_fp8, void const* scale1,
                          void const* scale2, void const* scale3, bool apply_weight) {
  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<OT>());
  torch::Tensor gating_tensor = torch::from_blob(
      gating_output, {static_cast<int64_t>(num_rows), static_cast<int64_t>(num_experts)}, origin_options);
  gating_tensor = gating_tensor.to(torch::kFloat32);

  llm_kernels::nvidia::RoutingFunctionType custom_routing_function;
  if (topk_method_ == "greedy" && scoring_func_ == "softmax") {
    custom_routing_function = llm_kernels::nvidia::RoutingFunctionType::GREEDY_TOPK_SOFTMAX_SCORE;
  } else if (topk_method_ == "fast" && scoring_func_ == "sigmoid") {
    custom_routing_function = llm_kernels::nvidia::RoutingFunctionType::FAST_TOPK_SIGMOID_SCORE;
  } else {
    KLLM_THROW(fmt::format("topk_method_ {} with scoring_func_ {} is not supported", topk_method_, scoring_func_));
  }

  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  moe_gemm.Gemm(input_activations, gating_tensor.data_ptr(), fc1_expert_weights, fc2_expert_weights, num_rows,
                hidden_size, inter_size, num_experts, topk, workspace_ptr, final_output, token_topk_final_scales,
                expanded_source_row_to_expanded_dest_row, expert_for_source_row, tp_size, rank, use_lora,
                best_config_index, NT, stream, is_fp8, scale1, scale2, scale3, custom_routing_function, apply_weight);
}

#define INVOKE_MOE_CUTLASS_GEMM(T, WT, OT, NT)                                                                         \
  template void InvokeMoeCutlassGemm<T, WT, OT, NT>(                                                                   \
      void const* input_activations, void* gating_output, void const* fc1_expert_weights,                              \
      void const* fc2_expert_weights, void* e_score_correction_bias, int64_t const num_rows,                           \
      int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const topk, char* workspace_ptr, \
      void* final_output, void* token_topk_final_scales, int* expanded_source_row_to_expanded_dest_row,                \
      int* expert_for_source_row, int tp_size, int rank, bool use_lora, size_t best_config_index, bool use_vllm_moe_,  \
      uint32_t num_expert_group_, uint32_t expert_groups_topk_, const std::string& scoring_func_,                      \
      const std::string& topk_method_, bool norm_topk_prob_, float routed_scaling_factor_,                             \
      bool use_e_score_correction_bias_, cudaStream_t stream, bool is_fp8, void const* scale1, void const* scale2,     \
      void const* scale3, bool apply_weight)

INVOKE_MOE_CUTLASS_GEMM(float, float, float, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(float, float, float, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
INVOKE_MOE_CUTLASS_GEMM(half, half, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(half, half, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#ifdef ENABLE_FP8
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#endif

#ifdef ENABLE_BFLOAT16
INVOKE_MOE_CUTLASS_GEMM(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#  ifdef ENABLE_FP8
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#  endif
#endif
#undef INVOKE_MOE_CUTLASS_GEMM

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L923
template <typename T>
void InvokeGroupedTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, int num_experts,
                       int topk, bool renormalize, int num_expert_group, int topk_group, std::string scoring_func,
                       void* e_bias, float routed_scaling_factor, int rank, cudaStream_t stream) {
  bool is_enable_fused_grouped_moe =
      (scoring_func == "sigmoid" && e_bias != nullptr && renormalize && topk < num_experts / num_expert_group);
  if (is_enable_fused_grouped_moe) {
    llm_kernels::nvidia::InvokeDeepSeekV3GroupedTopk<T>(gating_output, e_bias, routed_scaling_factor, topk_weights_ptr,
                                                        topk_ids_ptr, num_rows, num_experts, topk, num_expert_group,
                                                        topk_group, stream);
    return;
  }

  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
  torch::Tensor gating_tensor = torch::from_blob(
      gating_output, {static_cast<int64_t>(num_rows), static_cast<int64_t>(num_experts)}, origin_options);
  torch::Tensor scores;
  if (scoring_func == "softmax") {
    scores = torch::softmax(gating_tensor, -1);
  } else if (scoring_func == "sigmoid") {
    scores = gating_tensor.sigmoid();
  } else {
    KLLM_LOG_ERROR << fmt::format("Unsupported scoring function: {}", scoring_func);
  }

  torch::Tensor original_scores;
  torch::Tensor group_scores;
  int num_token = scores.size(0);
  if (e_bias != nullptr) {
    original_scores = scores.clone();
    auto fp32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
    torch::Tensor e_score_correction_bias_tensor = torch::from_blob(e_bias, {num_experts}, fp32_options);
    scores = scores + e_score_correction_bias_tensor;
    group_scores = std::get<0>(scores.view({num_token, num_expert_group, -1}).topk(2, -1)).sum(-1);
  } else {
    group_scores = std::get<0>(scores.view({num_token, num_expert_group, -1}).max(-1));
  }
  // KLLM_LOG_DEBUG << "group_scores " << group_scores;
  torch::Tensor group_idx = std::get<1>(torch::topk(group_scores, topk_group, -1, true));  // [n, top_k_group]
  torch::Tensor group_mask = torch::zeros_like(group_scores);                              // [n, n_group]
  group_mask.scatter_(1, group_idx, 1);                                                    // [n, n_group]
  torch::Tensor score_mask = group_mask.unsqueeze(-1)
                                 .expand({num_token, num_expert_group, scores.size(-1) / num_expert_group})
                                 .reshape({num_token, -1});  // [n, e]
  auto tmp_scores = scores.masked_fill(~score_mask.to(torch::kBool),
                                       -std::numeric_limits<float>::infinity());  // [n, e]

  torch::Tensor topk_weights, topk_ids;
  if (e_bias != nullptr) {
    topk_ids = std::get<1>(torch::topk(tmp_scores, topk, -1, true));
    // Use original unbiased scores for the routing weights
    topk_weights = original_scores.gather(1, topk_ids);
  } else {
    std::tie(topk_weights, topk_ids) = torch::topk(tmp_scores, topk, -1, true);
  }

  if (renormalize) {
    topk_weights = topk_weights / topk_weights.sum(-1, true);
  }
  // 这里做了提前乘routed_scaling_factor，
  // 如果要与vllm对比moe内部计算结果，
  // 需要将routed_scaling_factor设置为1.0。
  // routed_scaling_factor = 1.0;
  topk_weights = topk_weights * routed_scaling_factor;
  torch::Tensor output_topk_weights = topk_weights.cuda().to(torch::kFloat32).clone();
  torch::Tensor output_topk_ids = topk_ids.cuda().to(torch::kInt32).clone();

  CUDA_CHECK(cudaMemcpyAsync(topk_weights_ptr, output_topk_weights.data_ptr(), topk_weights.numel() * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(topk_ids_ptr, output_topk_ids.data_ptr(), topk_ids.numel() * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));
}
#define INVOKE_GROUPED_TOPK(T)                                                                                      \
  template void InvokeGroupedTopk<T>(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, \
                                     int num_experts, int topk, bool renormalize, int num_expert_group,             \
                                     int topk_group, std::string scoring_func, void* e_bias,                        \
                                     float routed_scaling_factor, int rank, cudaStream_t stream)
INVOKE_GROUPED_TOPK(float);
INVOKE_GROUPED_TOPK(half);
#ifdef ENABLE_BFLOAT16
INVOKE_GROUPED_TOPK(__nv_bfloat16);
#endif
#undef INVOKE_GROUPED_TOPK

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L624
template <typename T>
void InvokeFusedMoeKernelFunc(void* a, void* b, void* c, void* a_q, void* a_scale, void* b_scale, void* b_zp,
                              void* topk_weights, void* topk_ids, void* sorted_token_ids, void* expert_ids,
                              void* num_tokens_post_padded, bool mul_routed_weight, int topk, int chunk_num_tokens,
                              int numel, int m, int k, int n, int max_num_tokens_padded,
                              const std::unordered_map<std::string, int>& config, DataType weight_dtype,
                              DataType compute_dtype, bool use_moe_wna16_cuda, void* dequant_workspace,
                              std::vector<int>& block_shape, int num_experts, const cudaStream_t& stream) {
  if (chunk_num_tokens < config.at("block_size_m")) {
    max_num_tokens_padded = std::min(max_num_tokens_padded, chunk_num_tokens * topk * config.at("block_size_m"));
  }

  if (weight_dtype == DataType::TYPE_I4_GROUP && block_shape.size() > 1 && block_shape[1] > 0) {
    int group_size = block_shape[1];
    bool has_zp = !(b_zp == nullptr);
    int weight_bits = 4;

    std::unordered_map<std::string, int> moe_wna16_config = config;  // shallow copy
    UpdateMoeWna16BlockConfig(moe_wna16_config, use_moe_wna16_cuda,
                              compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3 && !has_zp, numel, k, n, n, group_size,
                              topk, moe_wna16_config.at("block_size_m"));

    if (use_moe_wna16_cuda) {
      void* topk_weights_input = mul_routed_weight ? topk_weights : nullptr;
      int num_token_blocks =
          (max_num_tokens_padded + moe_wna16_config.at("block_size_m") - 1) / moe_wna16_config.at("block_size_m");
      InvokeMoeWna16Gemm<T>(stream, c, a, b, b_scale, b_zp, topk_weights_input, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, topk, moe_wna16_config.at("block_size_m"),
                            moe_wna16_config.at("block_size_n"), moe_wna16_config.at("block_size_k"), weight_bits,
                            num_experts, m, n, k, group_size, num_token_blocks);
    } else if (compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3 && !has_zp) {  // TODO(jinxcwu) 目前只支持无zp的gptq
      // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L688
      // TODO(zakwang) :这里有个假定,即 block_shape 不为 None
      InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_scale, chunk_num_tokens, k, false, stream);
      DequantInt4Fp8(stream, dequant_workspace, b, (size_t)num_experts * n * k / 2);
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqInt4Fp8Kernel<T>(
          a_q, dequant_workspace, c, a_scale, b_scale, topk_weights, sorted_token_ids, expert_ids,
          num_tokens_post_padded, n, k, max_num_tokens_padded, numel, mul_routed_weight, topk, group_size,
          moe_wna16_config, stream);
    } else {
      int pack_factor = 8 / weight_bits;
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqAwqKernel<T>(
          a, b, c, b_scale, b_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k,
          max_num_tokens_padded, numel, k, 1, n * k / pack_factor, 1, k / pack_factor, n, 1, n * k / group_size, 1,
          k / group_size, n / pack_factor * k / group_size, 1, k / group_size, mul_routed_weight, topk, has_zp,
          weight_bits, group_size, moe_wna16_config, stream);
    }
  } else {
    if (compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
      // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L688
      // TODO(zakwang) :这里有个假定,即 block_shape 不为 None
      InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_scale, chunk_num_tokens, k, false, stream);
      a = a_q;
    }
    // A [m, k]
    // B [exprts_num, n, k]
    // C [m, topk, n]
    // A_scale [m, k / 128]
    // B_scale [experts_num, n / 128, k / 128]
    Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
        a, b, c, a_scale, b_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k,
        max_num_tokens_padded, numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128,
        block_shape[0], block_shape[1], mul_routed_weight, topk, compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3, false,
        config, stream);
  }
}

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L1271
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
                    cudaStream_t stream) {
  // hidden_states [num_tokens, hidden_size] dtype = T
  // gating_output [num_tokens, num_experts]
  // w1 [num_experts, inter_size * 2, hidden_size]
  // w2 [num_experts, hidden_size, inter_size]
  // topk_ids [num_tokens, topk]
  // topk_weights [num_tokens, topk]
  // M = min(num_tokens, CHUNK_SIZE)
  // N = inter_size * 2
  // intermediate_cache1 [M, topk, N]
  // intermediate_cache2 [M * topk, N // 2]
  // intermediate_cache3 [M, topk, hidden_size]
  // expert_map [num_experts]

  // a1_q, a2_q: FP8  需要将 FP16 输入量化成 FP8 + FLOAT32,这两个是两个对应空间
  // TODO(zezhao): 使用 num_experts / expert_para_size 来替换 total_num_experts. 不再维护 ExpertParallelSize
  size_t expert_para_size = Singleton<Environment>::GetInstance()->GetExpertParallelSize();
  size_t expert_world_size = Singleton<Environment>::GetInstance()->GetExpertWorldSize();
  int total_num_experts = num_experts * expert_para_size * expert_world_size;

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  cudaStream_t torch_stream = InvokeSetTorchStream(stream, rank);
#endif
  if (use_grouped_topk) {
    InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, topk,
                         renormalize, num_expert_group, topk_group, scoring_func_, e_bias, routed_scaling_factor, rank,
                         stream);
  } else {
    // 需要对非group的做优化，目前直接复用
    InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, topk,
                         renormalize, num_expert_group, topk_group, scoring_func_, e_bias, routed_scaling_factor, rank,
                         stream);
  }
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  InvokeSetTorchStream(torch_stream, rank);
#endif
  // Expert parallel.
  // hidden_state[num_tokens][hidden_dim]
  // topk_ids_ptr[num_tokens][topk]
  // topk_weights_ptr[num_tokens][topk]
  // token_node_map[node_ids][tokens_idx]

  // fused_experts_impl
  const int chunk_size = 32 * 1024;
  int M = std::min(num_tokens, chunk_size);
  int chunk_times = (num_tokens + chunk_size - 1) / chunk_size;

  std::unordered_map<std::string, int> config;
  bool use_moe_wna16_cuda = false;
  block_shape.resize(2);
  if (weight_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
    if (block_shape.size() < 2) {
      KLLM_LOG_ERROR << fmt::format("Config fp8_w8a8 need block_shape.shape = [2,]");
      return;
    }
    // This optimal configuration is obtained via deep_tune from
    // "/KsanaLLM/3rdparty/LLM_kernels/csrc/kernels/nvidia/fused_moe/fused_moe.py".
    // For details, refer to the best_config.json file in the same directory.
    // Usage instructions can be found in the README located in the same path.
    const int config_selection_threshold = 256;
    if (num_tokens > config_selection_threshold) {
      config = {{"block_size_m", 64},
                {"block_size_n", block_shape[0]},
                {"block_size_k", block_shape[1]},
                {"group_size_m", 1},
                {"num_warps", 4},
                {"num_stages", 3}};
    } else {
      config = {{"block_size_m", 16},
                {"block_size_n", block_shape[0]},
                {"block_size_k", block_shape[1]},
                {"group_size_m", 1},
                {"num_warps", 4},
                {"num_stages", 3}};
    }
  } else if (weight_dtype == DataType::TYPE_I4_GROUP && (!block_shape.empty())) {
    use_moe_wna16_cuda = ShouldMoeWna16UseCuda(M * topk, block_shape[1], num_experts, 4);
    if (use_moe_wna16_cuda) {
      config = {{"block_size_m", std::min(16, M)}};
    } else if (M <= 20) {
      config = {{"block_size_m", 16}, {"group_size_m", 1}};
    } else if (M <= 40) {
      config = {{"block_size_m", 32}, {"group_size_m", 1}};
    } else {
      config = {{"block_size_m", 64}, {"group_size_m", 1}};
    }
  } else {
    config = {{"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 8}};
    if (M <= num_experts || (M <= 32 && is_marlin)) {
      config = {{"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};
    }
  }
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
  // [start_offset0, length0, value0, ..., start_offsetN, lengthN, valueN]
  std::vector<int> fill_info;
  // reserve for fill_info
  const size_t fill_info_reserved_size = 24;
  fill_info.reserve(fill_info_reserved_size);
  for (int i = 0; i < chunk_times; ++i) {
    // curr_hidden_states [tokens_in_chunk, hidden_size]
    // curr_interm_cache1 [tokens_in_chunk, topk, inter_size * 2]
    // curr_interm_cache2 [tokens_in_chunk * topk, inter_size]
    // curr_interm_cache3 [tokens_in_chunk, topk, hidden_size]
    // curr_topk_ids [tokens_in_chunk, topk]
    // curr_topk_weights [tokens_in_chunk, topk]
    void* curr_hidden_states = hidden_states + sizeof(T) * chunk_size * i * hidden_size;
    int tokens_in_chunk = std::min(chunk_size, num_tokens - chunk_size * i);
    void* curr_topk_ids = topk_ids_ptr + sizeof(int32_t) * chunk_size * i * topk;
    void* curr_topk_weights = topk_weights_ptr + sizeof(float) * chunk_size * i * topk;
    void* curr_intermediate_cache1 = intermediate_cache1;
    void* curr_intermediate_cache2 = intermediate_cache2;
    void* curr_intermediate_cache3 = intermediate_cache3;

    void* fill_info_buffer_on_device = fused_id_buffer;
    void* curr_fused_id_buffer =
        fill_info_buffer_on_device + fill_info_reserved_size * sizeof(decltype(fill_info)::value_type);
    int id_buffer_offset = 0;

    // moe_align_block_size
    int block_size = config["block_size_m"];
    int numel = tokens_in_chunk * topk;
    int max_num_tokens_padded = numel + num_experts * (block_size - 1);
    // torch::Tensor sorted_ids = torch::empty({max_num_tokens_padded}, int32_options);
    // sorted_ids.fill_(numel);
    fill_info.insert(fill_info.end(), {id_buffer_offset, max_num_tokens_padded, numel});
    void* sorted_ids_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += max_num_tokens_padded;

    int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    // torch::Tensor expert_ids = torch::empty({max_num_m_blocks}, int32_options);
    // expert_ids.fill_(-1);
    fill_info.insert(fill_info.end(), {id_buffer_offset, max_num_m_blocks, -1});
    void* expert_ids_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += max_num_m_blocks;

    // torch::Tensor num_tokens_post_pad = torch::empty({1}, int32_options);
    fill_info.insert(fill_info.end(), {id_buffer_offset, 1, 0});
    void* num_tokens_post_pad_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += 1;
    // TODO(zezhao): SglMoe need support Expert-Parallel
    if (num_experts >= 224) {
      // torch::Tensor cumsum = torch::zeros({num_experts + 1}, int32_options);
      fill_info.insert(fill_info.end(), {id_buffer_offset, num_experts + 1, 0});
      void* cumsum_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
      id_buffer_offset += num_experts + 1;
      llm_kernels::nvidia::InvokeFillIntToBuffer(reinterpret_cast<int32_t*>(curr_fused_id_buffer),
                                                 fill_info_buffer_on_device, reinterpret_cast<int*>(fill_info.data()),
                                                 fill_info.size(), stream);
      llm_kernels::nvidia::InvokeSglMoeAlignBlockSize<int32_t>(
          reinterpret_cast<int32_t*>(curr_topk_ids), reinterpret_cast<int32_t*>(sorted_ids_ptr),
          reinterpret_cast<int32_t*>(expert_ids_ptr), reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), num_experts,
          block_size, numel, reinterpret_cast<int32_t*>(cumsum_ptr), stream);
    } else {
      // https://github.com/vllm-project/vllm/blob/185cc19f922a29868bf62e4f2674c763df36c8b5/csrc/moe/moe_align_sum_kernels.cu#L296
      llm_kernels::nvidia::InvokeFillIntToBuffer(reinterpret_cast<int32_t*>(curr_fused_id_buffer),
                                                 fill_info_buffer_on_device, reinterpret_cast<int*>(fill_info.data()),
                                                 fill_info.size(), stream);
      llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, UseExpertParallel>(
          reinterpret_cast<int32_t*>(curr_topk_ids), reinterpret_cast<int32_t*>(sorted_ids_ptr),
          reinterpret_cast<int32_t*>(expert_ids_ptr), reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), expert_map,
          topk, num_experts, expert_para_size, block_size, numel, rank, stream);
    }

    // invoke_fused_moe_kernel
    InvokeFusedMoeKernelFunc<T>(
        curr_hidden_states, w1, curr_intermediate_cache1, a1_q, a1_scale, w1_scale, w1_zp, curr_topk_weights,
        curr_topk_ids, reinterpret_cast<int32_t*>(sorted_ids_ptr), reinterpret_cast<int32_t*>(expert_ids_ptr),
        reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), false, topk, tokens_in_chunk, numel, tokens_in_chunk,
        hidden_size, inter_size * 2, max_num_tokens_padded, config, weight_dtype, compute_dtype, use_moe_wna16_cuda,
        dequant_workspace, block_shape, num_experts, stream);
    size_t elements_num = static_cast<size_t>(tokens_in_chunk) * topk * inter_size * 2;
    llm_kernels::nvidia::InvokeSiluAndMul<T, UseExpertParallel>(
        reinterpret_cast<const T*>(curr_intermediate_cache1), reinterpret_cast<T*>(curr_intermediate_cache2),
        reinterpret_cast<const int*>(curr_topk_ids), reinterpret_cast<const int*>(expert_map), num_experts,
        elements_num, inter_size, stream);

    InvokeFusedMoeKernelFunc<T>(
        curr_intermediate_cache2, w2, curr_intermediate_cache3, a2_q, a2_scale, w2_scale, w2_zp, curr_topk_weights,
        curr_topk_ids, reinterpret_cast<int32_t*>(sorted_ids_ptr), reinterpret_cast<int32_t*>(expert_ids_ptr),
        reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), true, 1, tokens_in_chunk * topk, numel,
        tokens_in_chunk * topk, inter_size, hidden_size, max_num_tokens_padded, config, weight_dtype, compute_dtype,
        use_moe_wna16_cuda, dequant_workspace, block_shape, num_experts, stream);

    void* curr_out_hidden_states = output_hidden_states + sizeof(T) * chunk_size * hidden_size * i;
    llm_kernels::nvidia::InvokeMoeSum<T, UseExpertParallel>(intermediate_cache3, curr_out_hidden_states, curr_topk_ids,
                                                            expert_map, tokens_in_chunk, num_experts, topk, hidden_size,
                                                            stream);
    fill_info.clear();
  }
}

#define FUSEDMOE(T)                                                                                                   \
  template void InvokeFusedMoe<T, true>(                                                                              \
      void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk, bool renormalize,      \
      const std::string& scoring_func_, void* e_bias, bool inplace, bool use_grouped_topk, int num_expert_group,      \
      int topk_group, DataType weight_dtype, DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, \
      void* w2_scale, void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,               \
      std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr, float routed_scaling_factor,          \
      void* output_hidden_states, void* intermediate_cache1, void* intermediate_cache2, void* intermediate_cache3,    \
      void* fused_id_buffer, int num_tokens, int num_experts, int hidden_size, int inter_size,                        \
      void* dequant_workspace, int rank, cudaStream_t stream);                                                        \
  template void InvokeFusedMoe<T, false>(                                                                             \
      void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk, bool renormalize,      \
      const std::string& scoring_func_, void* e_bias, bool inplace, bool use_grouped_topk, int num_expert_group,      \
      int topk_group, DataType weight_dtype, DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, \
      void* w2_scale, void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,               \
      std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr, float routed_scaling_factor,          \
      void* output_hidden_states, void* intermediate_cache1, void* intermediate_cache2, void* intermediate_cache3,    \
      void* fused_id_buffer, int num_tokens, int num_experts, int hidden_size, int inter_size,                        \
      void* dequant_workspace, int rank, cudaStream_t stream)
FUSEDMOE(float);
FUSEDMOE(half);
#ifdef ENABLE_BFLOAT16
FUSEDMOE(__nv_bfloat16);
#endif
#undef FUSEDMOE

}  // namespace ksana_llm
