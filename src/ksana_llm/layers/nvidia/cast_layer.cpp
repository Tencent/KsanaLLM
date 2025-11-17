/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status CastLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename SRC_DTYPE>
Status CastLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // When the number of input_tensors is greater than 1, perform a cast operation with an offset.
  // Set output_offset to the value of the first dimension of input_tensors[1].
  const auto& meta_data = input_tensors[1].shape;
  const size_t output_offset = meta_data[0];
  const size_t vocab_size = meta_data[1];
  const size_t vocab_size_pad = meta_data[2];

  const Tensor& input = input_tensors[0];
  Tensor& output = output_tensors[0];

  DataToFloat<SRC_DTYPE>(input.GetPtr<void>(), input.GetElementNumber(), vocab_size, vocab_size_pad,
                         output.GetPtr<void>() + output_offset, context_->GetComputeStreams()[rank_].Get());
  output.dtype = DataType::TYPE_FP32;
  output.shape = input.shape;
  return Status();
}

}  // namespace ksana_llm
