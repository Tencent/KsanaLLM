/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#if defined(ENABLE_VLLM_FLASH_ATTN_2) || defined(ENABLE_FLASH_ATTN_3)
#  include "ksana_llm/layers/set_torch_stream_layer.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status SetTorchStreamLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}
Status SetTorchStreamLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  torch_stream_ = InvokeSetTorchStream(this->context_->GetComputeStreams()[this->rank_].Get(), this->rank_);
  return Status();
}

void SetTorchStreamLayer::Clear() { torch_stream_ = InvokeSetTorchStream(torch_stream_, this->rank_); }

}  // namespace ksana_llm
#endif
