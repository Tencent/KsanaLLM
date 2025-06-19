/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status LayernormLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status LayernormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: input [token_num, hidden_size]
  //   1: weight [hidden_size]
  //   2: bias [hidden_size] (optional)
  // output_tensors:
  //   0: output [token_num, hidden_size]
  // Note: when bias is provided, compute layernorm, otherwise compute rmsnorm.
  const void* bias = input_tensors.size() > 2 ? input_tensors[2].GetPtr<void>() : nullptr;
  InvokeLayerNorm<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), bias, rms_norm_eps_,
                     input_tensors[0].shape[0], input_tensors[0].shape[1], output_tensors[0].GetPtr<void>(),
                     context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class LayernormLayer<float>;
template class LayernormLayer<half>;
#ifdef ENABLE_BFLOAT16
template class LayernormLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
