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
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_prepack_launcher.cuh
 */

#pragma once

#include <fmt/format.h>

#include "csrc/kernels/nvidia/machete/cutlass_extensions/torch_utils.hpp"
#include "csrc/kernels/nvidia/machete/machete_prepack_kernel.cuh"
#include "csrc/utils/nvidia/scalar_type.hpp"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

struct PrepackBArgs {
  const void* B_ptr;
  const std::vector<size_t>& B_shape;
  llm_kernels::nvidia::vllm_dtype::ScalarType a_type;
  llm_kernels::nvidia::vllm_dtype::ScalarType b_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_scales_type;
};

template <typename PrepackedLayoutB>
void prepack_impl(const void* inB, const std::vector<size_t> inB_shape, void* outB, cudaStream_t stream) {
  using ElementB = typename PrepackedLayoutB::ElementB;
  using PPBlockShape_NK = typename PrepackedLayoutB::PPBlockShape_NK;

  auto inB_ptr = static_cast<ElementB const*>(inB);
  auto outB_ptr = static_cast<ElementB*>(outB);
  // elements per storage item for B
  // auto eles_per_storage = (inB.dtype().itemsize() * 8) / cute::sizeof_bits_v<ElementB>;
  auto eles_per_storage = (sizeof(int32_t) * 8) / cute::sizeof_bits_v<ElementB>;

  // torch B passed in is/should be (packed_K,N), the kernel expects (N,K,L) (to
  // match cutlass using (N,K,L) for B), so we transpose B to (N,packed_K,L)
  // auto inBt_packed = inB.t();
  std::vector<size_t> inBt_packed_shape(inB_shape.rbegin(), inB_shape.rend());

  KLLM_KERNEL_CHECK_WITH_INFO(
      (inB_shape[0] * eles_per_storage) % size<1>(PPBlockShape_NK{}) == 0,
      fmt::format("B.shape[0] (in terms of unpacked elements) must be a multiple of {}", size<1>(PPBlockShape_NK{})));
  KLLM_KERNEL_CHECK_WITH_INFO(inB_shape[1] % size<0>(PPBlockShape_NK{}) == 0,
                              fmt::format("B.shape[1] must be a multiple of {}", size<0>(PPBlockShape_NK{})));

  using StrideB = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  // auto const l_inBt_packed = make_cute_layout<StrideB>(inBt_packed, "B");
  auto const l_inBt_packed = make_cute_layout_from_shape<StrideB>(inBt_packed_shape, "B");

  // convert (N,packed_K,L) layout to (N,K,L) layout
  //  in effect we want to do: blocked_product(layout_Bt_packed,
  //      make_ordered_layout(make_shape(_1{}, eles_per_storage, _1{}),
  //                          Step<_1, _0, _2>{}));
  // but blocked_product does not support dynamic strides so we implement the
  // equivalent manually,
  //   new_shape = (N, packed_K, L) * (1, eles_per_storage, 1) -> (N, K, L)
  //   new_stride = (s0, s1, s2) * (eles_per_storage, 1, eles_per_storage)
  //                 when s1 == 1
  KLLM_KERNEL_CHECK(stride<1>(l_inBt_packed) == 1);
  // clang-format off
  auto const layout_inBt = make_layout(
      transform_with_idx(l_inBt_packed.shape(), [&](auto ele, auto idx) {
        return idx == 1 ? ele * eles_per_storage : ele;
      }), 
      transform_with_idx(l_inBt_packed.stride(), [&](auto ele, auto idx) {
        return idx != 1 ? ele * eles_per_storage : ele;
      }));
  // clang-format on

  prepack_B_template<PrepackedLayoutB>(stream, inB_ptr, layout_inBt, outB_ptr);
};

void prepack_B_dispatch(PrepackBArgs args, void* outB, cudaStream_t stream);

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels