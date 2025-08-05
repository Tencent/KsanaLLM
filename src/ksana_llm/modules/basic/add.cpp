/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/add.h"

#include "ksana_llm/layers/add_layer.h"

namespace ksana_llm {

Add::Add(const LayerCreationContext& creation_context, const std::string& weight_name) {
  add_layer_ = std::make_shared<AddLayer>();
  add_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
  if (weight_name != "") {
    weight_ = creation_context.base_weight->GetModelWeights(weight_name);
    with_weight_ = true;
  }
}

Add::~Add() {}

Status Add::Forward(Tensor A, Tensor B, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_layer_->Forward({A, B}, output_tensors));
  return Status();
}

Status Add::Forward(Tensor A, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(add_layer_->Forward({A, weight_}, output_tensors));
  return Status();
}

}  // namespace ksana_llm
