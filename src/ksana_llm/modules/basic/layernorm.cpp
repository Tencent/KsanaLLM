/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/layernorm.h"

#include "ksana_llm/layers/layernorm_layer.h"

namespace ksana_llm {

Layernorm::Layernorm(const std::string& weight_name, float layernorm_eps,
                        const LayerCreationContext& creation_context, const std::string& weight_bias_name) {
  layernorm_layer_ = std::make_shared<LayernormLayer>();
  layernorm_layer_->Init({layernorm_eps}, creation_context.runtime_config, creation_context.context,
                         creation_context.rank);
  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  if (weight_bias_name != "") {
    weight_bias_ = creation_context.base_weight->GetModelWeights(weight_bias_name);
    with_bias_ = true;
  }
}

Layernorm::~Layernorm() {}

Status Layernorm::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (with_bias_) {
    // with bias weight
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({input_tensors[0], weight_, weight_bias_}, output_tensors));
  } else {
    STATUS_CHECK_RETURN(layernorm_layer_->Forward({input_tensors[0], weight_}, output_tensors));
  }
  return Status();
}

}  // namespace ksana_llm
