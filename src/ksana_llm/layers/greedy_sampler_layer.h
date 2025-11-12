/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

// A fast greedy sampler that uses local_argmax + all_gather + warp_argmax
// 1. Use local_argmax to get the local max and idx on each rank
// 2. Use all_gather to collect these max and idx from all ranks
// 3. Use warp_argmax to get the final argmax
class GreedySamplerLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensor, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensor, std::vector<Tensor>& output_tensors);
};

}  // namespace ksana_llm
