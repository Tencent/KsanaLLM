/*
 * Adapted from
 * https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/elementwise/concat_mla.cu
 *
 * Copyright 2023-2024 SGLang Team
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
 */

#include "csrc/kernels/nvidia/others/sglang/main/elementwise/concat_mla.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ void st_na_global_v1(const int* ptr, int v) {
  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void st_na_global_v2(const int2* ptr, const int2& v) {
  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ int ld_na_global_v1(const int* ptr) {
  int r;
  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
  return r;
}

__device__ __forceinline__ int2 ld_na_global_v2(const int2* ptr) {
  int2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
  return r;
}

constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

template <typename T, int HEAD_CHUNK_SIZE, int NUM_HEAD_CHUNKS>
__global__ void concat_mla_k_kernel(T* __restrict__ k, const T* __restrict__ k_nope, const T* __restrict__ k_rope,
                                    const int num_tokens) {
  constexpr int NUM_HEADS = HEAD_CHUNK_SIZE * NUM_HEAD_CHUNKS;
  constexpr int K_STRIDE_0 = NUM_HEADS * K_HEAD_DIM;
  constexpr int K_STRIDE_1 = K_HEAD_DIM;
  constexpr int K_NOPE_STRIDE_0 = NUM_HEADS * QK_NOPE_HEAD_DIM;
  constexpr int K_NOPE_STRIDE_1 = QK_NOPE_HEAD_DIM;
  constexpr int K_ROPE_STRIDE_0 = QK_ROPE_HEAD_DIM;

  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  if (token_id >= num_tokens) {
    return;
  }

  using NopeVec = int2;  // 8B/thread，32 thread = 256B/row
  using RopeVec = int;   // 4B/thread，32 thread = 128B/row
  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), "rope vec mismatch");

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  const int2* __restrict__ nope_src =
      reinterpret_cast<const int2*>(k_nope + token_id * K_NOPE_STRIDE_0 + head_row0 * K_NOPE_STRIDE_1) + lane_id;

  int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * K_STRIDE_0 + head_row0 * K_STRIDE_1) + lane_id;

  int* __restrict__ rope_dst =
      reinterpret_cast<int*>(k + token_id * K_STRIDE_0 + head_row0 * K_STRIDE_1 + QK_NOPE_HEAD_DIM) + lane_id;

  constexpr int NOPE_SRC_STRIDE_V = (K_NOPE_STRIDE_1 >> 2);  // int2 covers 4 bf16
  constexpr int NOPE_DST_STRIDE_V = (K_STRIDE_1 >> 2);
  constexpr int ROPE_DST_STRIDE_V = (K_STRIDE_1 >> 1);  // int covers 2 bf16

  const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * K_ROPE_STRIDE_0);
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);

  NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      const int2* next_src = nope_src + NOPE_SRC_STRIDE_V;
      next = ld_na_global_v2(next_src);
    }

    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);

    nope_src += NOPE_SRC_STRIDE_V;
    nope_dst += NOPE_DST_STRIDE_V;
    rope_dst += ROPE_DST_STRIDE_V;

    cur = next;
  }
}

template <typename T>
void concat_mla_k(const T* k_nope, const T* k_rope, T* k, const int num_tokens, const int num_heads,
                  const int qk_nope_head_dim, const int qk_rope_head_dim, cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(qk_nope_head_dim == QK_NOPE_HEAD_DIM, "Only support qk_nope_head_dim == %d",
                              QK_NOPE_HEAD_DIM);
  KLLM_KERNEL_CHECK_WITH_INFO(qk_rope_head_dim == QK_ROPE_HEAD_DIM, "Only support qk_rope_head_dim == %d",
                              QK_ROPE_HEAD_DIM);

  constexpr int num_warps_per_block = 32;
  const int block_size = num_warps_per_block * 32;  // 1024

#define LAUNCH_CONCAT_MLA_K_KERNEL(HEAD_CHUNK_SIZE, NUM_HEAD_CHUNKS)                 \
  do {                                                                               \
    const int grid_size = div_up(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block); \
    concat_mla_k_kernel<T, HEAD_CHUNK_SIZE, NUM_HEAD_CHUNKS>                         \
        <<<grid_size, block_size, 0, stream>>>(k, k_nope, k_rope, num_tokens);       \
  } while (0)

  switch (num_heads) {
    case 1:
      LAUNCH_CONCAT_MLA_K_KERNEL(1, 1);
      break;
    case 2:
      LAUNCH_CONCAT_MLA_K_KERNEL(2, 1);
      break;
    case 4:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 1);
      break;
    case 8:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 2);
      break;
    case 16:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 4);
      break;
    case 32:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 8);
      break;
    case 64:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 16);
      break;
    case 128:
      LAUNCH_CONCAT_MLA_K_KERNEL(4, 32);
      break;
    default:
      KLLM_KERNEL_THROW("Unsupported num_heads: %d", num_heads);
  }

#undef LAUNCH_CONCAT_MLA_K_KERNEL
}

#define CONCAT_MLA_K(T) \
  template void concat_mla_k<T>(const T*, const T*, T*, const int, const int, const int, const int, cudaStream_t);
CONCAT_MLA_K(float);
CONCAT_MLA_K(half);
CONCAT_MLA_K(__nv_bfloat16);
#undef CONCAT_MLA_K

}  // namespace nvidia
}  // namespace llm_kernels
