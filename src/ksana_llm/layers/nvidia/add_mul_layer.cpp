/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_mul_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status AddMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  // First parameter: scale1
  if (parameters.size() > 0) {
    scale1_ = std::any_cast<float>(parameters[0]);
  }

  // Second parameter: scale2
  if (parameters.size() > 1) {
    scale2_ = std::any_cast<float>(parameters[1]);
  }

  return Status();
}

Status AddMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
  return Status();
}

template <typename T>
Status AddMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // Set output tensor properties
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  const int m = static_cast<int>(input_tensors[0].shape[0]);
  const int n = static_cast<int>(input_tensors[0].shape[1]);
  cudaStream_t stream = context_->GetComputeStreams()[rank_].Get();

  auto input1 = reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  auto input2 = reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>());
  auto output = output_tensors[0].GetPtr<void>();

  InvokeMulThenAdd<T>(input1, input2, scale1_, scale2_, m, n, output, stream);

  return Status();
}

}  // namespace ksana_llm