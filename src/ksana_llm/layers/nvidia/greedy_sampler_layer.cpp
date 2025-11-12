/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/greedy_sampler_layer.h"

#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "ksana_llm/kernels/argmax.h"

namespace ksana_llm {

Status GreedySamplerLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status GreedySamplerLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors.size() == 2, "GreedySamplerLayer expects 2 input tensors");
  KLLM_CHECK_WITH_INFO(output_tensors.size() == 1, "GreedySamplerLayer expects 1 output tensor");
  KLLM_CHECK_WITH_INFO(input_tensors[1].shape[0] == 0, "GreedySamplerLayer expects `logits_offset == 0`");

  const Tensor& input = input_tensors[0];
  const size_t batch_size = input.shape[0];
  const size_t vocab_size = input_tensors[1].shape[1];
  const size_t vocab_size_pad = input_tensors[1].shape[2];
  Tensor& output = output_tensors[0];

  if (tp_size_ == 1) {
    KLLM_CHECK(input.shape[1] == vocab_size_pad);
    // Case 1: No tensor parallelism, do a standard argmax over the vocab dim
    // input: [batch_size, vocab_size]
    // output: [batch_size]
    ArgMax<T>(input.GetPtr<const T>(), batch_size, vocab_size, output.GetPtr<uint32_t>(),
              context_->GetComputeStreams()[rank_]);
    output.shape = {batch_size};
    output.dtype = TYPE_UINT32;
  } else if (input.shape[1] * static_cast<size_t>(tp_size_) == vocab_size_pad) {
    // Case 2: do a local argmax on each rank to get the local max and idx
    // Store both max and idx in float type
    // input: [batch_size, vocab_size_pad / tp_size]
    // output: [batch_size, 2] (max, idx)
    llm_kernels::nvidia::InvokeLocalArgMaxReduce(
        input.GetPtr<const T>(), batch_size,
        input.shape[1] - (rank_ + 1 == tp_size_ ? vocab_size_pad - vocab_size : 0ul), input.shape[1], rank_,
        output.GetPtr<float>(), context_->GetComputeStreams()[rank_].Get());
    output.shape = {batch_size, 2};
    output.dtype = TYPE_FP32;
  } else {
    // Case 3: do a warp argmax to get the final argmax across all ranks
    // input: [batch_size, 2 * tp_size] (max0, idx0, max1, idx1, ...)
    // output: [batch_size]
    KLLM_CHECK(input.dtype == TYPE_FP32);
    KLLM_CHECK(input.shape[1] == 2 * static_cast<size_t>(tp_size_));
    llm_kernels::nvidia::InvokeWarpArgMaxReduce(input.GetPtr<const float>(), batch_size, tp_size_,
                                                output.GetPtr<uint32_t>(), context_->GetComputeStreams()[rank_].Get());
    output.shape = {batch_size};
    output.dtype = TYPE_UINT32;
  }

  return Status();
}

}  // namespace ksana_llm
