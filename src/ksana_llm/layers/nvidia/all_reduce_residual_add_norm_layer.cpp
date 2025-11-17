/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"


namespace ksana_llm {

Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
  std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  rms_norm_weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  return Status();
}

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}


template <typename T>
Status AllReduceResidualAddNormLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                               std::vector<Tensor>& output_tensors) {
  int token_num = input_tensors[0].shape[0];
  const size_t hidden_dim = input_tensors[0].shape[1];
  void* input = input_tensors[0].GetPtr<void>();
  void* residual_in_ptr = input_tensors[1].GetPtr<void>();
  void* layernorm_weight_ptr = rms_norm_weight_.GetPtr<void>();
  void* norm_out_ptr = output_tensors[0].GetPtr<void>();

  if (!is_init_) {
    InitTrtAllReduceWorkspace(rank_, context_->ext->GetTrtAllReduceBuffers(),
                              context_->ext->GetTrtAllReduceFlags(),
                              context_->ext->GetTrtAllReduceWorkspaces(),
                              context_->GetComputeStreams()[rank_].Get());
    is_init_ = true;
  }
  // Fused the next three steps into one kernel operation
  // Step 1: tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0)
  // Step 2: adds->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer)
  // Step 3: layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0)
  RunTrtFusedAllReduceResidualNorm<T>(input, rank_, token_num, hidden_dim,
                                      context_->ext->GetTrtAllReduceWorkspaces(),
                                      layernorm_weight_ptr, rms_norm_eps_, residual_in_ptr, residual_in_ptr,
                                      norm_out_ptr, context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
