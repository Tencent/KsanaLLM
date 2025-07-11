/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/quant/machete_utils.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

#ifdef ENABLE_CUDA

template <typename T>
torch::Tensor MacheteUtils::PackWeight(torch::Tensor& weight, QuantMode quant_method) {
  llm_kernels::nvidia::vllm_dtype::ScalarType weight_type =
      (quant_method == QUANT_GPTQ) ? llm_kernels::nvidia::vllm_dtype::kU4B8 : llm_kernels::nvidia::vllm_dtype::kU4;
  torch::Tensor prepack_weight = torch::empty_like(weight);
  torch::Tensor weightT = weight.t().contiguous();
  InvokeMachetePrepackWeight(weightT.data_ptr(), {weightT.size(1), weightT.size(0)}, prepack_weight.data_ptr(),
                             GetMacheteDataType<T>(), weight_type, GetMacheteDataType<T>(),
                             context_->GetMemoryManageStreams()[rank_].Get());
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return prepack_weight;
}

template torch::Tensor MacheteUtils::PackWeight<float>(torch::Tensor& weight, QuantMode quant_method);
template torch::Tensor MacheteUtils::PackWeight<float16>(torch::Tensor& weight, QuantMode quant_method);
template torch::Tensor MacheteUtils::PackWeight<bfloat16>(torch::Tensor& weight, QuantMode quant_method);

#endif

}  // namespace ksana_llm