/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

MoE::MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
         const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(false) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
}

MoE::MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
         const std::string& e_score_correction_bias_weight_name, const LayerCreationContext& creation_context,
         MoeScaleNormMode moe_scale_norm_mode)
    : use_e_score_correction_bias_(true) {
  Init(up_gate_proj_weight_name, down_proj_weight_name, creation_context, moe_scale_norm_mode);
  e_score_correction_bias_weight_ = creation_context.base_weight->GetModelWeights(e_score_correction_bias_weight_name);
}

void MoE::Init(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
               const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode) {
  moe_layer_ = creation_context.moe_layer_factory->AutoCreateMoeLayer(
      creation_context.base_weight, std::vector<std::string>{up_gate_proj_weight_name, down_proj_weight_name},
      creation_context.weight_type, creation_context.input_type, creation_context.output_type, {moe_scale_norm_mode});

  moe_layer_->SetWorkSpaceBuffer(creation_context.workspace_mgr->GetWorkspace(moe_layer_->GetWorkSpaceSize()));
  moe_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

  up_gate_proj_weight_ = creation_context.base_weight->GetModelWeights(up_gate_proj_weight_name);
  down_proj_weight_ = creation_context.base_weight->GetModelWeights(down_proj_weight_name);
}

MoE::~MoE() {}

Status MoE::Forward(Tensor hidden_states, Tensor gating_output, Tensor workspace_tensor,
                    std::vector<Tensor>& output_tensors) {
  std::vector<Tensor> moe_output_tensors = {output_tensors[0], workspace_tensor};
  if (use_e_score_correction_bias_) {
    STATUS_CHECK_RETURN(moe_layer_->Forward(
        {hidden_states, gating_output, up_gate_proj_weight_, down_proj_weight_, e_score_correction_bias_weight_},
        moe_output_tensors));
  } else {
    STATUS_CHECK_RETURN(moe_layer_->Forward({hidden_states, gating_output, up_gate_proj_weight_, down_proj_weight_},
                                            moe_output_tensors));
  }
  output_tensors[0].shape = moe_output_tensors[0].shape;
  return Status();
}

}  // namespace ksana_llm
