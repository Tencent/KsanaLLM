/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/split.h"

#include "ksana_llm/layers/split_layer.h"

namespace ksana_llm {

Split::Split(const LayerCreationContext& creation_context) {
  split_layer_ = std::make_shared<SplitLayer>();
  split_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

Split::~Split() {}

Status Split::Forward(Tensor input, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(split_layer_->Forward({input}, output_tensors));
  return Status();
}

}  // namespace ksana_llm
