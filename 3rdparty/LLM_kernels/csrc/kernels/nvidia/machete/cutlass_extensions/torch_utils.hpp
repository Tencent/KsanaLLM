/*
 * Copyright 2025 vLLM Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/tree/v0.8.2/csrc/cutlass_extensions
 */

#pragma once

#include "cute/layout.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

namespace cute {

namespace detail {

template <class T, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr auto tapply_with_idx(T&& t, F&& f, G&& g,
                                                seq<I...>) {
  return g(f(cute::get<I>(static_cast<T&&>(t)), I)...);
  
}

template <class F, int... I>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f, seq<I...>) {
  return make_shape(f(I)...);
}

};  // namespace detail

template <class T, class F>
CUTE_HOST_DEVICE constexpr auto transform_with_idx(T const& t, F&& f) {
  if constexpr (cute::is_tuple<T>::value) {
    return detail::tapply_with_idx(
        t, f, [](auto const&... a) { return cute::make_tuple(a...); },
        tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// calls: make_shape(f(0), f(1), ..., f(N-1))
template <int N, class F>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f) {
  return detail::make_shape_from_idx(f, make_seq<N>{});
}

};  // namespace cute

// Make a layout from a tensor with `rank(Stride{})`, where the shape is the
// shape of the passed in tensor and the strides are of type `Stride` and
// contain the strides of the passed in tensor, checking that any static strides
// in `Stride{}` match the strides of the passed in tensor.
// If `tensor.dim() < rank(Stride{})`, the shape is padded with 1s and the extra
// strides are set to be 0 or 1.
template <typename Stride>
static inline auto make_cute_layout_from_shape(const std::vector<size_t>& tensor_shape,
                                    std::string_view name = "tensor") {
  int tensor_dim = tensor_shape.size();
  KLLM_KERNEL_CHECK(tensor_dim <= rank(Stride{}));

  // Assuming a contiguous layout for stride calculation
  std::vector<size_t> tensor_stride(tensor_dim);
  size_t current_stride = 1;
  for (int i = tensor_dim - 1; i >= 0; --i) {
    tensor_stride[i] = current_stride;
    current_stride *= tensor_shape[i];
  }

  auto stride = cute::transform_with_idx(
      Stride{}, [&](auto const& stride_ele, auto const& idx) {
        using StrideEle = std::decay_t<decltype(stride_ele)>;

        if (idx < tensor_dim) {
          if constexpr (cute::is_static_v<StrideEle>) {
            KLLM_KERNEL_CHECK_WITH_INFO(StrideEle::value == tensor_stride[idx], fmtstr("Expected {}.stride({}) to be {}",
                        name, idx, StrideEle::value));
            return StrideEle{};
          } else {
            if (tensor_shape[idx] == 1) {
              return StrideEle{0};
            } else {
              return StrideEle{static_cast<long>(tensor_stride[idx])};
            }
          }
        } else {
          if constexpr (cute::is_static_v<StrideEle>) {
            static_assert(StrideEle::value == 0 || StrideEle::value == 1);
          }
          return StrideEle{};
        }
      });

  auto shape = cute::make_shape_from_idx<rank(Stride{})>([&](auto const& idx) {
    if (idx < tensor_dim)
      return static_cast<int64_t>(tensor_shape[idx]);
    else
      return int64_t(1);
  });

  return make_layout(shape, stride);
}

template <typename Stride>
static inline auto maybe_make_cute_layout_from_shape(
    std::optional<std::vector<size_t>> const& tensor_shape,
    std::string_view name = "tensor") {
  using Layout = decltype(make_cute_layout_from_shape<Stride>(*tensor_shape));

  if (tensor_shape) {
    return std::optional<Layout>{make_cute_layout_from_shape<Stride>(*tensor_shape, name)};
  } else {
    return std::optional<Layout>{};
  }
}
