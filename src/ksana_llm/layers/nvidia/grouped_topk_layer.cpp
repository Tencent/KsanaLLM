/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/kernels/nvidia/moe_kernel_wrapper.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

Status GroupedTopkLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  int parameter_index = 0;
  topk_ = std::any_cast<int>(parameters[parameter_index++]);
  renormalize_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<int>(parameters[parameter_index++]);
  topk_group_ = std::any_cast<int>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);

  expert_para_size_ = runtime_config.parallel_basic_config.expert_parallel_size;
  expert_world_size_ = runtime_config.parallel_basic_config.expert_world_size;

  return Status();
}

Status GroupedTopkLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status GroupedTopkLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  // [0] gating_output [num_tokens, num_experts]
  // [1] e_bias (optional, can be nullptr)
  //
  // output_tensors:
  // [0] topk_weights_ptr [num_tokens, topk]
  // [1] topk_ids_ptr [num_tokens, topk]

  void* gating_output = input_tensors[0].GetPtr<void>();
  void* e_bias = nullptr;

  // 根据 use_e_score_correction_bias_ 和输入张量数量决定是否使用 e_bias
  if (use_e_score_correction_bias_ && input_tensors.size() > 1) {
    e_bias = input_tensors[1].GetPtr<void>();
  }

  void* topk_weights_ptr = output_tensors[0].GetPtr<void>();
  void* topk_ids_ptr = output_tensors[1].GetPtr<void>();

  int num_tokens = input_tensors[0].shape[0];
  int num_experts = input_tensors[0].shape[1];

  // 计算 total_num_experts，考虑专家并行
  // TODO(zezhao): 使用 num_experts / expert_para_size 来替换 total_num_experts. 不再维护 ExpertParallelSize
  int total_num_experts = num_experts;

  InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, topk_,
                       renormalize_, num_expert_group_, topk_group_, scoring_func_, e_bias, routed_scaling_factor_,
                       rank_, context_->GetComputeStreams()[rank_].Get());

  return Status();
}

}  // namespace ksana_llm
