/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class Linear {
 public:
  // Disable a default constructor
  Linear(const std::string& weight_name, const LayerCreationContext& creation_context,
         const GroupQuantBackend& group_quant_backend);

  ~Linear();
  Status Forward(Tensor input_tensor, std::vector<Tensor>& output_tensors);

  // TODO(robertyuan): Remove later
  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> proj_layer_;
  Tensor weight_;
};
}  // namespace ksana_llm
