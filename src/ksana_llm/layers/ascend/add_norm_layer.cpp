/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_norm_layer.h"

namespace ksana_llm {

Status AddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
  std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  auto layernorm_eps = std::any_cast<float>(parameters[parameter_index++]);
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps}, runtime_config, context, rank);
  add_layer_ = std::make_shared<AddLayer>();
  add_layer_->Init({}, runtime_config, context, rank);
  return Status();
}

Status AddNormLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  std::vector<Tensor> add_output_tensors = {input_tensors[1]};
  STATUS_CHECK_RETURN(add_layer_->Forward({input_tensors[0], input_tensors[1]}, add_output_tensors));
  STATUS_CHECK_RETURN(layernorm_layer_->Forward({input_tensors[1], input_tensors[2]}, output_tensors));
  return Status();
}

}  // namespace ksana_llm
