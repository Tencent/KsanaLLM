/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/add_norm.h"
#include "ksana_llm/layers/add_norm_layer.h"
namespace ksana_llm {

AddNorm::AddNorm(const std::string& weight_name, float norm_eps, const LayerCreationContext& creation_context) {
  add_norm_layer_ = std::make_shared<AddNormLayer>();
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  add_norm_layer_->Init({norm_eps}, creation_context.runtime_config, creation_context.context,
                        creation_context.rank);
}

AddNorm::~AddNorm() {}

Status AddNorm::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors[0]: hidden_buffer_tensors_0[0]
  // input_tensors[1]: residual_buffer[0]
  // output_tensors[0]: hidden_buffer_tensors_0[0]
  STATUS_CHECK_RETURN(add_norm_layer_->Forward({input_tensors[0], input_tensors[1], weight_}, output_tensors));
  return Status();
}


}  // namespace ksana_llm
