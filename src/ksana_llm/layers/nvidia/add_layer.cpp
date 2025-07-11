/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status AddLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  return Status();
}

template <typename T>
Status AddLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto a = reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  auto b = reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>());
  // The Add-Bias-Residual Kernel uses the shape[0] of the input tensor to determine whether
  // broadcasting is required.
  if (input_tensors[0].shape[0] == input_tensors[1].shape[0]) {
    InvokeAddBiasResidual<T>(a, b, nullptr, static_cast<int>(input_tensors[0].shape[0]),
                             static_cast<int>(input_tensors[0].shape[1]), output_tensors[0].GetPtr<void>(),
                             context_->GetComputeStreams()[rank_].Get());
  } else {
    InvokeAddBiasResidual<T>(a, nullptr, b, static_cast<int>(input_tensors[0].shape[0]),
                             static_cast<int>(input_tensors[0].shape[1]), output_tensors[0].GetPtr<void>(),
                             context_->GetComputeStreams()[rank_].Get());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class AddLayer<float>;
template class AddLayer<half>;
template class AddLayer<__nv_bfloat16>;

}  // namespace ksana_llm
