/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/mul_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status MulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status MulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int input_0_m = input_tensors[0].shape[0];
  int input_0_n = input_tensors[0].shape[1];
  int input_1_m = input_tensors[1].shape[0];
  int input_1_n = input_tensors[1].shape[1];
  if (input_0_m != input_1_m && input_0_m != 1 && input_1_m != 1) {
    KLLM_THROW(
        fmt::format("The size of first tensor {} must match the size of second tensor {} at non-singleton dimension 0",
                    input_0_m, input_1_m));
  }
  if (input_0_n != input_1_n && input_0_n != 1 && input_1_n != 1) {
    KLLM_THROW(
        fmt::format("The size of first tensor {} must match the size of second tensor {} at non-singleton dimension 1",
                    input_0_n, input_1_n));
  }
  InvokeMul<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), output_tensors[0].GetPtr<void>(),
               input_0_m, input_0_n, input_1_m, input_1_n, rank_);
  size_t output_m = static_cast<size_t>((input_0_m >= input_1_m) ? input_0_m : input_1_m);
  size_t output_n = static_cast<size_t>((input_0_n >= input_1_n) ? input_0_n : input_1_n);
  output_tensors[0].shape = {output_m, output_n};
  output_tensors[0].dtype = input_tensors[1].dtype;
  return Status();
}

}  // namespace ksana_llm
