/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/linear.h"

namespace ksana_llm {

Linear::Linear(const std::string& weight_name, const LayerCreationContext& creation_context,
               const LinearComputeBackend& linear_compute_backend, const bool skip_quant) {
  proj_layer_ = creation_context.matmul_layer_factory->AutoCreateLayer(
      creation_context.base_weight, weight_name, creation_context.weight_type, creation_context.input_type,
      creation_context.output_type, linear_compute_backend, std::vector<std::any>{skip_quant});
  proj_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(proj_layer_->GetWorkspaceSize()));

  // TODO(robertyuan): Merge Proprocess and Init
  proj_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

#ifdef ENABLE_ACL
  proj_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
#endif

  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

Linear::~Linear() {}

Status Linear::Forward(Tensor input_tensor, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(proj_layer_->Forward({input_tensor, weight_}, output_tensors));
  return Status();
}

Status Linear::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Forward(input_tensors[0], output_tensors);
}

}  // namespace ksana_llm
