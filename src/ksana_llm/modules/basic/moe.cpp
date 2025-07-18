/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

template <typename T>
MoE<T>::MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
            const LayerCreationContext<T>& creation_context, MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(false) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
}

template <typename T>
MoE<T>::MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
            const std::string& e_score_correction_bias_weight_name, const LayerCreationContext<T>& creation_context,
            MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(true) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
  e_score_correction_bias_weight_ = creation_context.base_weight->GetModelWeights(e_score_correction_bias_weight_name);
}

template <typename T>
void MoE<T>::Init(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
                  const LayerCreationContext<T>& creation_context, MoeScaleNormMode moe_scale_norm_mode) {
  moe_layer_ = creation_context.matmul_layer_factory->AutoCreateMoeLayer(
      creation_context.base_weight, std::vector<std::string>{up_gate_proj_weight_name, down_proj_weight_name},
      creation_context.weight_type, creation_context.input_type, creation_context.output_type, {moe_scale_norm_mode});

  up_gate_proj_weight_ = creation_context.base_weight->GetModelWeights(up_gate_proj_weight_name);
  down_proj_weight_ = creation_context.base_weight->GetModelWeights(down_proj_weight_name);
  expert_map_ = creation_context.base_weight->GetModelWeights("expert_map");
}

template <typename T>
MoE<T>::~MoE() {}

template <typename T>
Status MoE<T>::Forward(Tensor hidden_states, Tensor gating_output, std::vector<Tensor>& output_tensors) {
  if (use_e_score_correction_bias_) {
    STATUS_CHECK_RETURN(moe_layer_->Forward({hidden_states, gating_output, up_gate_proj_weight_, down_proj_weight_,
                                             expert_map_, e_score_correction_bias_weight_},
                                            output_tensors));
  } else {
    STATUS_CHECK_RETURN(moe_layer_->Forward(
        {hidden_states, gating_output, up_gate_proj_weight_, down_proj_weight_, expert_map_}, output_tensors));
  }
  return Status();
}

template class MoE<float>;
template class MoE<float16>;
template class MoE<bfloat16>;

}  // namespace ksana_llm
