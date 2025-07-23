/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class EcclAllGatherLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
};

}  // namespace ksana_llm
