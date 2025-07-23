/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_VLLM_FLASH_ATTN_2
#  include <torch/torch.h>
#  include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

template <typename T>
class SetTorchStreamLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
  void Clear() override;

#  ifdef ENABLE_CUDA
 private:
  cudaStream_t torch_stream_;
#  endif
};

}  // namespace ksana_llm
#endif
