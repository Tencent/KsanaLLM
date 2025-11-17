/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"
#include "ksana_llm/models/communicator/tp_communicator.h"

namespace ksana_llm {

Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
  std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  rms_norm_weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  tp_comm_ = std::any_cast<std::shared_ptr<TpCommunicator>>(parameters[parameter_index++]);
  return Status();
}

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  return Status(RET_NOT_IMPLEMENTED, "AllReduceResidualAddNormLayer overloaded forward is not implemented");
}

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors, const bool is_multi_token_forward,
                                              ForwardingContext& forwarding_context) {
  std::vector<Tensor> reduce_input = {input_tensors[0]};
  tp_comm_->AllReduce(reduce_input, output_tensors, is_multi_token_forward, forwarding_context);
  add_norm_layer_->Forward({output_tensors[0], input_tensors[1], rms_norm_weight_}, output_tensors);
  return Status();
}

}  // namespace ksana_llm
