/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>
#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class InputRefitLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
  void Clear() { cast_tensor_vec_.clear(); }

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 public:
  std::vector<torch::Tensor> cast_tensor_vec_;
};

}  // namespace ksana_llm
