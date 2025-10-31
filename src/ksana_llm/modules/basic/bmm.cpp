/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/bmm.h"

#include "ksana_llm/layers/batched_matmul_layer.h"

namespace ksana_llm {

Bmm::Bmm(const std::string& weight_name, const LayerCreationContext& creation_context,
         const LinearComputeBackend& linear_compute_backend) {
  bmm_layer_ = creation_context.matmul_layer_factory->AutoCreateLayer(
      creation_context.base_weight, "", TYPE_VOID, creation_context.input_type, creation_context.output_type,
      linear_compute_backend, {});
  bmm_layer_->SetWorkspaceBuffer(creation_context.workspace_mgr->GetWorkspace(bmm_layer_->GetWorkspaceSize()));

  bmm_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

  context_ = creation_context.context;
  rank_ = creation_context.rank;

  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

Status Bmm::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors.size() == 1, "should have exactly one input tensor.");
  KLLM_CHECK_WITH_INFO(output_tensors.size() == 1, "should have exactly one output tensor.");

  const Tensor& input_tensor = input_tensors[0];

  if (input_tensor.shape[1] == weight_.shape[0] && input_tensor.shape[2] == weight_.shape[1]) {
    return bmm_layer_->Forward({input_tensor, weight_}, output_tensors);
  }

  KLLM_THROW("The input shapes: " + Vector2Str(std::vector<size_t>(input_tensors[0].shape)) + " and weight shapes: " +
             Vector2Str(std::vector<size_t>(weight_.shape)) + " of bmm that have not been implemented yet.");
  return Status();
}

}  // namespace ksana_llm
