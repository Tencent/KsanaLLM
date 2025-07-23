/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/split_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status SplitLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                           std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status SplitLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  std::vector<T*> output_ptrs(output_tensors.size());
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    output_ptrs[i] = output_tensors[i].GetPtr<T>();
  }
  std::vector<int> col_offsets(output_tensors.size() + 1, 0);
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    col_offsets[i + 1] = col_offsets[i] + output_tensors[i].shape[1];
  }
  InvokeSplit<T>(input_tensors[0].GetPtr<T>(), output_ptrs, col_offsets, input_tensors[0].shape[0],
                 input_tensors[0].shape[1], output_tensors.size(), context_->GetComputeStreams()[rank_].Get());
  return Status();
}

template class SplitLayer<float>;
template class SplitLayer<half>;
template class SplitLayer<__nv_bfloat16>;

}  // namespace ksana_llm
