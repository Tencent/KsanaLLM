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
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_prepack_kernel.cuh
 */

#pragma once

#include "csrc/kernels/nvidia/machete/cutlass_extensions/cute_utils.cuh"
#include "csrc/kernels/nvidia/machete/cutlass_extensions/torch_utils.hpp"
#include "csrc/kernels/nvidia/machete/machete_mm_kernel.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

template <int threads, typename PrepackedLayoutB, typename BInTensor, typename ElementB>
static __global__ void prepack_B_kernel(BInTensor B_in, ElementB* B_out_ptr) {
  auto constexpr block_size = Int<size(typename PrepackedLayoutB::PPBlockShape_NK{})>{};
  auto constexpr eles_per_thread = Int<block_size / threads>{};
  static_assert(block_size % threads == 0, "block_size must be divisible by the number of threads");

  // Which pre-packed are we responsible for
  auto blk_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.z);
  auto tB_in = local_tile(B_in, append(typename PrepackedLayoutB::PPBlockShape_NK{}, _1{}), blk_coord);

  // Find the start offset in the output for this pre-packed block
  auto bNbKL_to_offset = PrepackedLayoutB::bNbKL_to_offset(shape(B_in));

  // Tensor representing a 1:1 mapping to the output space in 1D
  auto tB_out_linear =
      make_tensor(get_logical_ptr(B_out_ptr) + bNbKL_to_offset(blk_coord), make_layout(make_shape(block_size)));
  // Mapping from output space (1D) to input space
  auto tB_in_linear =
      make_tensor(tB_in.data(), tB_in.layout()
                                    .compose(right_inverse(PrepackedLayoutB::ppblock_ilvd_NK_to_offset()))
                                    .with_shape(make_shape(block_size)));

  // Tile for this specific thread (could have used a TiledCopy but these work
  // best with 2d layouts, this is a simple 1d layout so local_tile is enough,
  // we are also not that concerned with performance for this kernel)
  auto thr_tB_in_linear = local_tile(tB_in_linear, make_shape(eles_per_thread), threadIdx.x);
  auto thr_tB_out_linear = local_tile(tB_out_linear, make_shape(eles_per_thread), threadIdx.x);

  // Construct a register-backed Tensor with the same shape as each thread's
  // partition
  auto fragment = make_tensor<ElementB>(shape(thr_tB_in_linear));

  copy(thr_tB_in_linear, fragment);
  copy(Copy_Atom<DefaultCopy, uint8_t>{}, fragment, thr_tB_out_linear);
}

template <typename PrepackedLayoutB, typename InLayout>
static void prepack_B_template(cudaStream_t stream, typename PrepackedLayoutB::ElementB const* B_in_ptr,
                               InLayout B_layout, typename PrepackedLayoutB::ElementB* B_out_ptr) {
  using TileShapeNKL = decltype(append(typename PrepackedLayoutB::PPBlockShape_NK{}, _1{}));

  KLLM_KERNEL_CHECK(size<0>(B_layout) % size<0>(TileShapeNKL{}) == 0);
  KLLM_KERNEL_CHECK(size<1>(B_layout) % size<1>(TileShapeNKL{}) == 0);

  auto N_tiles = size<0>(B_layout) / size<0>(TileShapeNKL{});
  auto K_tiles = size<1>(B_layout) / size<1>(TileShapeNKL{});
  auto L_tiles = size<2>(B_layout);

  auto B_in = make_tensor(get_logical_ptr(B_in_ptr), B_layout);

  prepack_B_kernel<128, PrepackedLayoutB><<<dim3(N_tiles, K_tiles, L_tiles), 128, 0, stream>>>(B_in, B_out_ptr);
}

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels