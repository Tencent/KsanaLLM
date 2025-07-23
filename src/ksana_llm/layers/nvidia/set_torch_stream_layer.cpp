/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_VLLM_FLASH_ATTN_2
#  include "ksana_llm/layers/set_torch_stream_layer.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status SetTorchStreamLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename T>
Status SetTorchStreamLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  torch_stream_ = InvokeSetTorchStream(this->context_->GetComputeStreams()[this->rank_].Get(), this->rank_);
  return Status();
}

template <typename T>
void SetTorchStreamLayer<T>::Clear() {
  torch_stream_ = InvokeSetTorchStream(torch_stream_, this->rank_);
}

template class SetTorchStreamLayer<float>;
template class SetTorchStreamLayer<half>;
template class SetTorchStreamLayer<__nv_bfloat16>;

}  // namespace ksana_llm
#endif
