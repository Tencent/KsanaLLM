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
  void* output_ptr = output_tensors[0].GetPtr<void>();
  // When the number of input_tensors is greater than 1, perform a cast operation with an offset.
  // Set output_offset to the value of the first dimension of input_tensors[1].
  size_t output_offset = input_tensors[1].shape[0];
  output_ptr += output_offset;
  DataToFloat<SRC_DTYPE>(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                         input_tensors[0].GetElementNumber(), input_tensors[1].shape[1], input_tensors[1].shape[2],
                         output_ptr, context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].dtype = DataType::TYPE_FP32;
  output_tensors[0].shape = input_tensors[0].shape;
  return Status();
}

}  // namespace ksana_llm
