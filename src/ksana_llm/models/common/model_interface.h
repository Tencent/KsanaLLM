/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"

namespace ksana_llm {
template <typename T>
class ModelInterface {
 public:
  ModelInterface() {}
  ~ModelInterface() = default;

  virtual Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) = 0;

  virtual Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) = 0;

  virtual Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) = 0;
};
}  // namespace ksana_llm
