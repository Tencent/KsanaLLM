/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "ksana_llm/layers/marlin_moe_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

Status MarlinMoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_vllm_moe_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  expert_groups_topk_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  topk_method_ = std::any_cast<std::string>(parameters[parameter_index++]);
  norm_topk_prob_ = std::any_cast<bool>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);
  enable_full_shared_expert_ = std::any_cast<bool>(parameters[parameter_index++]);
  DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  group_size_ = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);
  return Status();
}

size_t MarlinMoeLayer::GetWorkSpaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkSpaceSizeT); }

template <typename T>
size_t MarlinMoeLayer::GetWorkSpaceSizeT() {
  max_gating_size_ = sizeof(float) * max_token_num_ * expert_num_;
  max_ws_bytes_ = max_gating_size_;
  max_ws_bytes_ += InvokeGetFusedMarlinMoeWorkspaceSize(max_token_num_, expert_inter_size_, expert_hidden_size_,
                                                        expert_num_, expert_topk_, sizeof(T));
  return max_ws_bytes_;
}

Status MarlinMoeLayer::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  return Status();
}

Status MarlinMoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors[0].dtype == TYPE_FP16) {
    half* output = static_cast<half*>(output_tensors[0].GetPtr<void>());
    half* input = static_cast<half*>(input_tensors[0].GetPtr<void>());

    void* w1 = input_tensors[2].GetPtr<void>();
    void* w2 = input_tensors[3].GetPtr<void>();
    if (w1 == nullptr || w2 == nullptr) {
      KLLM_THROW("w1 or w2 is nullptr.");
    }

    void* w1_scales = input_tensors[2].scales->GetPtr<void>();
    void* w2_scales = input_tensors[3].scales->GetPtr<void>();
    if (w1_scales == nullptr || w2_scales == nullptr) {
      KLLM_THROW("w1_scales or w2_scales is nullptr.");
    }

    int* w1_g_idx = (input_tensors[2].g_idx ? static_cast<int*>(input_tensors[2].g_idx->GetPtr<void>()) : nullptr);
    int* w2_g_idx = (input_tensors[3].g_idx ? static_cast<int*>(input_tensors[3].g_idx->GetPtr<void>()) : nullptr);

    void* w1_zeros = (input_tensors[2].zeros ? input_tensors[2].zeros->GetPtr<void>() : nullptr);
    void* w2_zeros = (input_tensors[3].zeros ? input_tensors[3].zeros->GetPtr<void>() : nullptr);

    int* w1_perm = (input_tensors[2].perm ? static_cast<int*>(input_tensors[2].perm->GetPtr<void>()) : nullptr);
    int* w2_perm = (input_tensors[3].perm ? static_cast<int*>(input_tensors[3].perm->GetPtr<void>()) : nullptr);
    half* gating_output = static_cast<half*>(input_tensors[1].GetPtr<void>());
    float* gating_output_float = static_cast<float*>(workspace_buffer_->GetPtr<void>());
    int gating_output_size = input_tensors[1].shape[0] * input_tensors[1].shape[1];

    DataToFloat<half>(gating_output, gating_output_size, 1, 1, gating_output_float,
                      context_->GetComputeStreams()[rank_].Get());

    void* workspace = workspace_buffer_->GetPtr<void>() + max_gating_size_;
    size_t workspace_size = max_ws_bytes_ - max_gating_size_;

    int num_tokens = input_tensors[0].shape[0];
    llm_kernels::nvidia::MOEExpertScaleNormalizationMode norm_mode =
        static_cast<llm_kernels::nvidia::MOEExpertScaleNormalizationMode>(static_cast<int>(moe_scale_norm_mode_));
    FusedMarlinMoe(output, input, gating_output_float, w1, w2, w1_scales, w2_scales, workspace, workspace_size,
                   num_tokens, expert_inter_size_, expert_hidden_size_, expert_num_, expert_topk_,
                   context_->GetComputeStreams()[rank_].Get(), norm_mode, w1_g_idx, w2_g_idx, w1_perm, w2_perm,
                   w1_zeros, w2_zeros, num_bits_, group_size_);
    output_tensors[0].shape = input_tensors[0].shape;
    output_tensors[0].dtype = input_tensors[0].dtype;
  } else {
    KLLM_THROW("marlin moe only support input type " + std::to_string(TYPE_FP16) + ", not support input type " +
               std::to_string(input_tensors[0].dtype));
  }
  return Status();
}


}  // namespace ksana_llm
