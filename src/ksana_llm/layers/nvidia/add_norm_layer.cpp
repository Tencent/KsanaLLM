/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_norm_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status AddNormLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  KLLM_LOG_DEBUG << fmt::format("rms_norm_eps {}", rms_norm_eps_);
  return Status();
}

template <typename T>
Status AddNormLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors[0].GetPtr<void>() != output_tensors[0].GetPtr<void>()) {
    KLLM_THROW("AddNormLayer input[0] and output[0] tensors must be the same because it is an inplace operation.");
  }
  // input_tensors:
  //   0: input [token_num, hidden_size] modified in-place
  //   1: residual [token_num, hidden_size] modified in-place
  //   2: weight [hidden_size]
  // Step 1: residual[i] += input[i]
  // Step 2: input[i] = (residual[i] / RMS(residual)) * weight[i]
  InvokeFusedAddRmsNorm<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(),
                           input_tensors[2].GetPtr<void>(), rms_norm_eps_, input_tensors[0].shape[0],
                           input_tensors[0].shape[1], context_->GetComputeStreams()[rank_].Get());

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class AddNormLayer<float>;
template class AddNormLayer<half>;
template class AddNormLayer<__nv_bfloat16>;

}  // namespace ksana_llm
