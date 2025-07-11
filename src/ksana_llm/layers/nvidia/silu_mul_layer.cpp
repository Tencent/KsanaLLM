/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status SiluMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  return Status();
}

template <typename T>
Status SiluMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors.size() == 1) {
    InvokeRowBasedGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
        reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), static_cast<int>(input_tensors[0].shape[0]),
        static_cast<int>(input_tensors[0].shape[1]), output_tensors[0].GetPtr<void>(),
        context_->GetComputeStreams()[rank_].Get());
    output_tensors[0].shape = input_tensors[0].shape;
    output_tensors[0].shape[1] = output_tensors[0].shape[1] / 2;
    output_tensors[0].dtype = input_tensors[0].dtype;
  } else if (input_tensors.size() == 2) {
    InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
        reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), /* bias */ nullptr,
        reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), /* gated_bias */ nullptr,
        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
    output_tensors[0].shape = input_tensors[0].shape;
    output_tensors[0].dtype = input_tensors[0].dtype;
  }

  return Status();
}

template class SiluMulLayer<float>;
template class SiluMulLayer<half>;
template class SiluMulLayer<__nv_bfloat16>;

}  // namespace ksana_llm
