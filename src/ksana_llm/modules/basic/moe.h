/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class MoE {
 public:
  // Disable a default constructor
  MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
      const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode);

  MoE(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
      const std::string& e_score_correction_bias_weight_name, const LayerCreationContext& creation_context,
      MoeScaleNormMode moe_scale_norm_mode);

  ~MoE();

  Status Forward(Tensor hidden_states, Tensor gating_output, Tensor workspace_tensor,
                 std::vector<Tensor>& output_tensors);

 private:
  void Init(const std::string& up_gate_proj_weight_name, const std::string& down_proj_weight_name,
            const LayerCreationContext& creation_context, MoeScaleNormMode moe_scale_norm_mode);

 protected:
  std::shared_ptr<BaseLayer> moe_layer_;
  Tensor up_gate_proj_weight_;
  Tensor down_proj_weight_;

  bool use_e_score_correction_bias_;
  Tensor e_score_correction_bias_weight_;
};
}  // namespace ksana_llm
