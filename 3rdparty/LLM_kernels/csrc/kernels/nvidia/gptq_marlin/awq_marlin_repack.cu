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
 * https://github.com/vllm-project/vllm/tree/65334ef3b9e4fd32ebc5c4e512debc25d5025488/csrc/quantization/gptq_marlin
 */

#include <fmt/format.h>

#include "csrc/kernels/nvidia/gptq_marlin/marlin.cuh"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace marlin {

template <int const num_threads, int const num_bits>
__global__ void awq_marlin_repack_kernel(uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
                                         int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int tile_n_ints = tile_n_size / pack_factor;

  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;
    int first_n_packed = first_n / pack_factor;

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * tile_k_size;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) + first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    // Undo interleaving
    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];

      int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
      int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) + sh_stride * cur_elem];

      vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
      vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id, n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

#define CALL_IF(NUM_BITS)                                                                                      \
  else if (num_bits == NUM_BITS) {                                                                             \
    cudaFuncSetAttribute(marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>,                   \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);                         \
    marlin::awq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS>                                         \
        <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(b_q_weight_ptr, out_ptr, size_k, size_n); \
  }

void awq_marlin_repack(const uint32_t* b_q_weight_ptr, uint32_t* out_ptr, int64_t size_k, int64_t size_n,
                       int64_t num_bits, int rank, cudaStream_t stream) {
  int blocks = 0, max_shared_mem = 0;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, rank);
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, rank);
  KLLM_KERNEL_CHECK(max_shared_mem > 0);

  if (false) {
  }
  CALL_IF(4)
  CALL_IF(8)
  else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, fmt::format("Unsupported repack config: num_bits = {}", num_bits));
  }
}

std::vector<int64_t> awq_marlin_repack_meta(int64_t size_k, int64_t size_n, int64_t num_bits) {
  int const pack_factor = 32 / num_bits;
  return {size_k / marlin::tile_size, size_n * marlin::tile_size / pack_factor};
}

}  // namespace marlin
}  // namespace nvidia
}  // namespace llm_kernels