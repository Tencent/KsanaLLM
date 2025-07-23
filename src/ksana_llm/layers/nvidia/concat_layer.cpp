/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/concat_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status ConcatLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  concat_dim = std::any_cast<const size_t>(parameters[0]);
  return Status();
}

template <typename T>
Status ConcatLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const auto& input_a = input_tensors[0];
  const auto& input_b = input_tensors[1];
  auto& output = output_tensors[0];

  // check shape
  std::vector<size_t> input_a_shape = std::vector<size_t>(input_a.shape);
  std::vector<size_t> input_b_shape = std::vector<size_t>(input_b.shape);
  KLLM_CHECK_WITH_INFO(input_a_shape.size() == input_b_shape.size(), "shape size must be identical in cacat");
  for (size_t i = 0; i < input_a_shape.size(); ++i) {
    if (i == concat_dim) {
      continue;
    }
    KLLM_CHECK_WITH_INFO(
        input_a_shape[i] == input_b_shape[i],
        fmt::format("The shapes of all dimensions except the concatenation dimension must be identical. {} vs {}",
                    input_a.ToString(), input_b.ToString()));
  }

  output.dtype = input_a.dtype;
  output.shape = input_a.shape;
  output.shape[concat_dim] = input_a_shape[concat_dim] + input_b_shape[concat_dim];

  std::vector<size_t> output_shape = std::vector<size_t>(output.shape);

  const size_t inner_dim_size =
      std::accumulate(output_shape.begin() + concat_dim + 1, output_shape.end(), 1, std::multiplies<>());
  const size_t outer_dim_size =
      std::accumulate(output_shape.begin(), output_shape.begin() + concat_dim, 1, std::multiplies<>());
  Concat<T>(reinterpret_cast<const void*>(input_a.GetPtr<void>()),
            reinterpret_cast<const void*>(input_b.GetPtr<void>()), input_a_shape[concat_dim], input_b_shape[concat_dim],
            outer_dim_size, inner_dim_size, reinterpret_cast<void*>(output.GetPtr<void>()),
            context_->GetComputeStreams()[rank_].Get());

  return Status();
}

template class ConcatLayer<float>;
template class ConcatLayer<half>;
template class ConcatLayer<__nv_bfloat16>;

}  // namespace ksana_llm
