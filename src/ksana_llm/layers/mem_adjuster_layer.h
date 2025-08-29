/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class MemAdjusterLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  // Extract a subset of columns from a 2D tensor
  // Input: tensor of shape (m, n)
  // Output: tensor[:, input_offset:input_offset + output_n] with shape (m, output_n)
  Status ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset, size_t output_n);

 private:
  template <typename T>
  Status ExtractSubMatrixT(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset, size_t output_n);
};

}  // namespace ksana_llm
