/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/bmm.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/permute_layer.h"

namespace ksana_llm {

Bmm::Bmm(const std::string& weight_name, const LayerCreationContext& creation_context,
         const GroupQuantBackend& group_quant_backend) {
  bmm_layer_ = creation_context.matmul_layer_factory->AutoCreateLayer(
      creation_context.base_weight, "", TYPE_VOID, creation_context.input_type, creation_context.output_type,
      group_quant_backend, {});

  bmm_layer_->SetWorkSpaceBuffer(creation_context.workspace_mgr->GetWorkspace(bmm_layer_->GetWorkSpaceSize()));
  bmm_layer_->Preprocess(creation_context.model_config, creation_context.runtime_config);

  context_ = creation_context.context;
  rank_ = creation_context.rank;

  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

Status Bmm::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // weight shape = [heads_num_per_tp:16, qk_nope_head_dim:128, kv_lora_rank:512]
  auto heads_num_per_tp = weight_.shape[0];
  auto qk_nope_head_dim = weight_.shape[1];
  KLLM_CHECK_WITH_INFO(input_tensors.size() >= 1, "should have one input tensor.");

  if (input_tensors.size() >= 3 && input_tensors[0].shape[1] == heads_num_per_tp &&
      input_tensors[0].shape[2] == qk_nope_head_dim) {
    // last two inputs are temp tensors
    Tensor& permute1_in = const_cast<Tensor&>(input_tensors[0]);
    Tensor permute1_out = const_cast<Tensor&>(input_tensors[1]);
    Permute(permute1_in, permute1_out, {1, 0, 2}, context_->GetComputeStreams()[rank_]);
    // tp8: [tokens, heads, qk_nope_dims]: [256, 16, 128] => [16, 256, 128]
    permute1_out.shape = {permute1_in.shape[1], permute1_in.shape[0], permute1_in.shape[2]};

    std::vector<Tensor> bmm_out = {input_tensors[2]};
    // [heads, tokens, qk_nope_head_dim] * [heads, qk_nope_head_dim, kv_lora_rank]
    // [16, 256, 128] * [16, 128, 512] => [16, 256, 512]
    STATUS_CHECK_RETURN(bmm_layer_->Forward({permute1_out, weight_}, bmm_out));

    Tensor& permute2_in = bmm_out[0];
    Tensor& permute2_out = output_tensors[0];
    // [16, 256, 512] => [256, 16, 512]
    Permute(permute2_in, permute2_out, {1, 0, 2}, context_->GetComputeStreams()[rank_]);
    permute2_out.shape = {permute2_in.shape[1], permute2_in.shape[0], permute2_in.shape[2]};
    return Status();
  }
  KLLM_THROW("The input shapes: " + Vector2Str(std::vector<size_t>(input_tensors[0].shape)) + " and weight shapes: " +
             Vector2Str(std::vector<size_t>(weight_.shape)) + " of bmm that have not been implemented yet.");
  return Status();
}

}  // namespace ksana_llm
