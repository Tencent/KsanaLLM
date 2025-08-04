/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class Layernorm {
 public:
  // Disable a default constructor
  Layernorm(const std::string& weight_name, float layernorm_eps, const LayerCreationContext& creation_context,
            const std::string& weight_bias_name = "");

  ~Layernorm();

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  std::shared_ptr<BaseLayer> layernorm_layer_;
  Tensor weight_;
  Tensor weight_bias_;
  bool with_bias_ = false;
};
}  // namespace ksana_llm
