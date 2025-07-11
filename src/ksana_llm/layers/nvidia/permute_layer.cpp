/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/permute_layer.h"

#include "ksana_llm/kernels/permute.h"

namespace ksana_llm {

template <typename T>
std::vector<size_t> PermuteLayer<T>::FindPermutation(const std::vector<size_t>& input_shape,
                                                     const std::vector<size_t>& output_shape) {
  std::vector<size_t> permutation;
  std::vector<bool> used(input_shape.size(), false);
  for (int dim_size : output_shape) {
    bool found = false;
    for (int i = 0; i < input_shape.size(); ++i) {
      if (input_shape[i] == dim_size && !used[i]) {
        permutation.push_back(i);
        used[i] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      KLLM_THROW(fmt::format("Element in output_shape: {} not found in input_shape: {}", Vector2Str(output_shape),
                             Vector2Str(input_shape)));
    }
  }

  return permutation;
}

template <typename T>
Status PermuteLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors[0].shape.size() == output_tensors[0].shape.size(),
                       "Permute layer need have same size shape.");
  std::vector<size_t> permutation = FindPermutation(static_cast<std::vector<size_t>>(input_tensors[0].shape),
                                                    static_cast<std::vector<size_t>>(output_tensors[0].shape));
  KLLM_LOG_DEBUG << "Permute layer permutation: " << Vector2Str(permutation);
  Permute(const_cast<Tensor&>(input_tensors[0]), output_tensors[0], permutation, context_->GetComputeStreams()[rank_]);
  return Status();
}

template class PermuteLayer<float>;
template class PermuteLayer<half>;
template class PermuteLayer<__nv_bfloat16>;

}  // namespace ksana_llm