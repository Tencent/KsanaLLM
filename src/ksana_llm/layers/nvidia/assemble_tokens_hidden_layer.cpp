/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status AssembleTokensHiddenLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status AssembleTokensHiddenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  const int accepted_tokens_size = input_tensors[1].shape[0];
  AssembleTokensHidden<T>(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                          reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()), accepted_tokens_size,
                          input_tensors[0].shape[1], reinterpret_cast<void*>(output_tensors[0].GetPtr<void>()),
                          context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].shape[0] = accepted_tokens_size;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class AssembleTokensHiddenLayer<float>;
template class AssembleTokensHiddenLayer<half>;
template class AssembleTokensHiddenLayer<__nv_bfloat16>;

}  // namespace ksana_llm
