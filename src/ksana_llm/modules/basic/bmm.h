/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class Bmm {
 public:
  // Disable a default constructor
  Bmm(const std::string& weight_name, const LayerCreationContext& creation_context,
      const GroupQuantBackend& group_quant_backend);

  ~Bmm() = default;

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> bmm_layer_;
  Tensor weight_;

  int rank_;
  std::shared_ptr<Context> context_;
};
}  // namespace ksana_llm