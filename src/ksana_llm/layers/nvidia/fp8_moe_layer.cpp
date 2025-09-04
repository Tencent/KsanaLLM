/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/layers/fp8_moe_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {
#define WT fp8e4m3  // TODO(robertyuan): Only support fp8e4m3 rightnow.

Status Fp8MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  return MoeLayer::Init(parameters, runtime_config, context, rank);
}

size_t Fp8MoeLayer::GetWorkSpaceSize() { DISPATCH_BY_2_DTYPE(inter_data_type_, GetWorkSpaceSizeT); }

Status Fp8MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, PreprocessT, model_config, runtime_config);
}

Status Fp8MoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
size_t Fp8MoeLayer::GetWorkSpaceSizeT() {
  GetMoeGemmWorkspaceSize<WT, WT, T>(this->max_token_num_, this->expert_num_per_node_, this->expert_hidden_size_,
                                     this->expert_inter_size_, this->expert_topk_, this->tp_size_, this->rank_,
                                     this->use_lora_, this->max_ws_bytes_, this->workspace_info_.workspace_sizes);
  quant_buffer_size_ = this->max_token_num_ * this->expert_hidden_size_ * GetTypeSize(TYPE_FP8_E4M3);
  this->max_ws_bytes_ += quant_buffer_size_;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for Fp8MoeLayer", this->rank_, this->max_ws_bytes_);
  return this->max_ws_bytes_;
}

Status Fp8MoeLayer::SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  this->workspace_buffer_ = workspace_buffer;
  this->scale_probabilities_size_ = this->max_token_num_ * this->expert_num_per_node_ * sizeof(float);
  this->src_to_dest_map_size_ = this->expert_topk_ * this->max_token_num_ * sizeof(int);
  this->selected_expert_size_ = this->expert_topk_ * this->max_token_num_ * sizeof(int);
  this->lora_workspace_size_ = 0;  // NO support for lora
  this->moe_workspace_size_ = this->max_ws_bytes_ - this->scale_probabilities_size_ - this->src_to_dest_map_size_ -
                              this->selected_expert_size_ - this->lora_workspace_size_ - quant_buffer_size_;
  return Status();
}

template <typename T>
Status Fp8MoeLayer::PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  bool is_fp8 = true;
  best_config_index_ = InvokeMoeGemmConfigProfile<WT, WT, T>(this->tactics_, is_fp8);
  return Status();
}

template <typename T>
Status Fp8MoeLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts
  const size_t num_tokens = input_tensors[0].shape[0];
  bool is_fp8 = true;
  void* fc1_dequant_scale = nullptr;
  void* fc2_quant_scale = nullptr;
  void* fc2_dequant_scale = nullptr;
  std::shared_ptr<Tensor> buffer = this->workspace_buffer_;
  void* input_activations = buffer->GetPtr<void>();

  if (this->set_workspace_buffer_info_) {
    this->set_workspace_buffer_info_ = false;

    this->workspace_info_.size = this->max_ws_bytes_ - quant_buffer_size_;
    this->workspace_info_.workspace = buffer->GetPtr<void>() + quant_buffer_size_;
    this->workspace_info_.scale_probs = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.workspace), this->moe_workspace_size_);
    this->workspace_info_.src_to_dest_map = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.scale_probs), this->scale_probabilities_size_);
    this->workspace_info_.selected_experts = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.src_to_dest_map), this->src_to_dest_map_size_);
    this->workspace_info_.lora_workspace = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(this->workspace_info_.selected_experts), this->selected_expert_size_);
  }

  if (input_tensors[2].weight_scales && input_tensors[2].input_scales) {
    float* fc1_quant_scale = static_cast<float*>(input_tensors[2].input_scales->GetPtr<void>());
    // convert input to fp8
    Fp8E4m3Quantize<T>(1, input_tensors[0].GetElementNumber(), static_cast<T*>(input_tensors[0].GetPtr<void>()),
                       input_activations, fc1_quant_scale, true,
                       this->context_->GetComputeStreams()[this->rank_].Get());
    fc1_dequant_scale = input_tensors[2].weight_scales->GetPtr<void>();
    fc2_quant_scale = input_tensors[3].input_scales->GetPtr<void>();
    fc2_dequant_scale = input_tensors[3].weight_scales->GetPtr<void>();
  } else {
    KLLM_THROW("Unsupported fp8 moe method, only support both weight_scale and input_scale are not null.");
  }
  void* e_score_correction_bias_weight_void = nullptr;
  if (this->use_e_score_correction_bias_) {
    e_score_correction_bias_weight_void = input_tensors[4].GetPtr<void>();
  }
  if (this->moe_scale_norm_mode_ == MoeScaleNormMode::RE_NORM) {
    InvokeMoeCutlassGemm<WT, WT, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE>(
        input_activations, input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
        input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, this->expert_hidden_size_,
        this->expert_inter_size_, this->expert_num_per_node_, this->expert_topk_, this->workspace_info_.workspace_sizes,
        static_cast<char*>(this->workspace_info_.workspace), output_tensors[0].GetPtr<void>(),
        this->workspace_info_.scale_probs, static_cast<int*>(this->workspace_info_.src_to_dest_map),
        static_cast<int*>(this->workspace_info_.selected_experts), this->tp_size_, this->rank_, this->use_lora_,
        best_config_index_, this->tactics_, this->use_vllm_moe_, this->num_expert_group_, this->expert_groups_topk_,
        this->scoring_func_, this->topk_method_, this->norm_topk_prob_, this->routed_scaling_factor_,
        this->use_e_score_correction_bias_, this->context_->GetComputeStreams()[this->rank_].Get(), is_fp8,
        fc1_dequant_scale, fc2_quant_scale, fc2_dequant_scale);
  } else if (this->moe_scale_norm_mode_ == MoeScaleNormMode::NO_NORM) {
    InvokeMoeCutlassGemm<WT, WT, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE>(
        input_activations, input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
        input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, this->expert_hidden_size_,
        this->expert_inter_size_, this->expert_num_per_node_, this->expert_topk_, this->workspace_info_.workspace_sizes,
        static_cast<char*>(this->workspace_info_.workspace), output_tensors[0].GetPtr<void>(),
        this->workspace_info_.scale_probs, static_cast<int*>(this->workspace_info_.src_to_dest_map),
        static_cast<int*>(this->workspace_info_.selected_experts), this->tp_size_, this->rank_, this->use_lora_,
        best_config_index_, this->tactics_, this->use_vllm_moe_, this->num_expert_group_, this->expert_groups_topk_,
        this->scoring_func_, this->topk_method_, this->norm_topk_prob_, this->routed_scaling_factor_,
        this->use_e_score_correction_bias_, this->context_->GetComputeStreams()[this->rank_].Get(), is_fp8,
        fc1_dequant_scale, fc2_quant_scale, fc2_dequant_scale);
  }

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

#undef WT

}  // namespace ksana_llm
