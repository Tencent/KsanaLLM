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
 * https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_interleaving_utils.cuh
 */

#pragma once

#include "cute/layout.hpp"
#include "cutlass/cutlass.h"

namespace llm_kernels {
namespace nvidia {
namespace machete {

using namespace cute;

// get an interleaved block layout where each element consecutive element has a
// stride of bit_stride and the block width is blk_bit_width,
// examples:
//  size_bits<T> = 8, bit_stride = 8,  blk_bit_width = 32 -> 4:1
//  size_bits<T> = 8, bit_stride = 16, blk_bit_width = 32 -> (2, 2):(2, 1)
//  size_bits<T> = 4, bit_stride = 8,  blk_bit_width = 32 -> (4, 2):(2, 1)
//  size_bits<T> = 4, bit_stride = 16, blk_bit_width = 32 -> (2, 4):(4, 1)
template <typename T, int bit_stride, int blk_bit_width>
CUTE_HOST_DEVICE static constexpr auto get_interleaved_blk_layout() {
  static_assert(blk_bit_width % bit_stride == 0);
  static_assert(bit_stride % cute::sizeof_bits_v<T> == 0);

  constexpr auto elems_per_blk = blk_bit_width / cute::sizeof_bits_v<T>;

  if constexpr (cute::sizeof_bits_v<T> == bit_stride) {
    // identity layout
    return Layout<Shape<Int<elems_per_blk>>>{};
  } else {
    constexpr auto elems_per_stride = bit_stride / cute::sizeof_bits_v<T>;
    constexpr auto num_strides = elems_per_blk / elems_per_stride;
    return Layout<Shape<Int<num_strides>, Int<elems_per_stride>>, Stride<Int<elems_per_stride>, Int<1>>>{};
  }
}

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
