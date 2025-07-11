/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/mixture_of_experts/moe_kernels.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::nvidia;
using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T, typename WT, typename OT>
void MoeGemmWrapper<T, WT, OT>::GetWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size,
                                                 size_t expert_inter_size, size_t expert_topk, int tp_size, int rank,
                                                 bool use_lora, size_t& ws_bytes) {
  llm_kernels::nvidia::MOEParallelismConfig parallelism_config(tp_size, rank, 1, 0);
  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, WT, OT>>();
  size_t moe_workspace_size =
      moe_gemm->getWorkspaceSize(token_num, expert_hidden_size, expert_inter_size, expert_num, expert_topk,
                                 llm_kernels::nvidia::ActivationType::Swiglu, parallelism_config, use_lora);
  // Output of post-softmax routing probabilities
  size_t scale_probabilities_size = token_num * expert_num * sizeof(float);
  // Permutation map
  size_t src_to_dest_map_size = expert_topk * token_num * sizeof(int);
  // Selected expert map
  size_t selected_expert_size = expert_topk * token_num * sizeof(int);
  size_t lora_workspace_size = 0;
  if (use_lora) {
    // TODO(winminkong): add lora workspace size
  }
  ws_bytes =
      moe_workspace_size + scale_probabilities_size + src_to_dest_map_size + selected_expert_size + lora_workspace_size;
}

template <typename T, typename WT, typename OT>
size_t MoeGemmWrapper<T, WT, OT>::GetBestConfigIndex(bool is_fp8) {
  size_t best_config_index = 0;
  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, WT, OT>>();
  std::vector<cutlass_extensions::CutlassGemmConfig> configs;

  int sm = GetSMVersion();
  auto tactics = moe_gemm->getFilteredTactics(sm, is_fp8);
  bool is_sm90 = sm >= 90;
  // TODO(winminkong): profile and select best config index
  auto it = std::find_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { return c.is_sm90 == is_sm90; });
  if (it != tactics.end()) {
    best_config_index = std::distance(tactics.begin(), it);
  }

  return best_config_index;
}

template <typename T, typename WT, typename OT>
void MoeGemmWrapper<T, WT, OT>::Gemm(void const* input_activations_void, void const* gating_output,
                                     void const* fc1_expert_weights_void, void const* fc2_expert_weights_void,
                                     int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                     int const num_experts, int const topk, char* workspace_ptr,
                                     void* final_output_void, void* token_topk_final_scales_void,
                                     int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row,
                                     int tp_size, int rank, bool use_lora, size_t best_config_index,
                                     MOEExpertScaleNormalizationMode moe_norm_mode, cudaStream_t stream, bool is_fp8,
                                     void const* scale1_void, void const* scale2_void, void const* scale3_void,
                                     RoutingFunctionType custom_routing_function, bool apply_weight) {
  int64_t const num_not_finished = num_rows;
  llm_kernels::nvidia::MOEParallelismConfig parallelism_config(tp_size, rank, 1, 0);
  QuantParams quant_params{};
  if (is_fp8 && scale1_void && scale2_void && scale3_void) {
    quant_params = QuantParams::FP8(static_cast<float const*>(scale1_void), static_cast<float const*>(scale2_void),
                                    static_cast<float const*>(scale3_void));
  }
  // TODO(winminkong): support lora moe
  LoraParams lora_params{};

  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, WT, OT>>();
  std::vector<cutlass_extensions::CutlassGemmConfig> configs = moe_gemm->getFilteredTactics(GetSMVersion(), is_fp8);
  moe_gemm->setTactic(configs[best_config_index], configs[best_config_index]);

  // mixtral : MOEExpertScaleNormalizationMode::RENORMALIZE
  // qwen2_moe : MOEExpertScaleNormalizationMode::NONE
  moe_gemm->runMoe(input_activations_void, static_cast<float const*>(gating_output), fc1_expert_weights_void, nullptr,
                   llm_kernels::nvidia::ActivationType::Swiglu, fc2_expert_weights_void, nullptr, quant_params,
                   num_rows, hidden_size, inter_size, num_experts, topk, workspace_ptr, final_output_void, nullptr,
                   num_not_finished, token_topk_final_scales_void, expanded_source_row_to_expanded_dest_row,
                   expert_for_source_row, parallelism_config, moe_norm_mode, use_lora, lora_params, stream,
                   custom_routing_function, apply_weight);
}

template class MoeGemmWrapper<float, float, float>;
template class MoeGemmWrapper<half, half, half>;
#ifdef ENABLE_FP8
template class MoeGemmWrapper<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
#endif
template class MoeGemmWrapper<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
#  ifdef ENABLE_FP8
template class MoeGemmWrapper<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
#  endif

}  // namespace nvidia
}  // namespace llm_kernels
