/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
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
 *
 * Adapted from
 * [SGLang Project] https://github.com/sgl-project/sglang/blob/03886917bd59f12a1420a99150997732ffea52da/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu#L19
 */

#include "per_token_group_quant_8bit.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_BFLOAT16
#  include <cuda_bf16.h>
#endif
#ifdef ENABLE_FP8
#  include <cuda_fp8.h>
#endif
#include <cub/cub.cuh>

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"

namespace llm_kernels {
namespace nvidia {

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

#ifdef ENABLE_FP8

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false>
__global__ void per_token_group_quant_8bit_kernel(const T* __restrict__ input, void* __restrict__ output_q,
                                                  float* __restrict__ output_s, const int group_size,
                                                  const int num_groups, const int groups_per_block, const float eps,
                                                  const float min_8bit, const float max_8bit,
                                                  const int scale_num_rows = 0, const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  const int block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  float* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int row_idx = global_group_id / scale_num_rows;
    const int col_idx = global_group_id % scale_num_rows;
    scale_output = output_s + (col_idx * scale_stride + row_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#  pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  const float y_s = local_absmax / max_8bit;

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#  pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
      group_output[i * vec_size + j] = DST_DTYPE(q_val);
    }
  }
}

template <typename T>
void per_token_group_quant_8bit(void* input, void* output_q, void* output_s, int m, int n, int64_t group_size,
                                    float eps, float min_8bit, float max_8bit, bool is_column_major,
                                    cudaStream_t stream) {
  const int num_groups = m * n / group_size;

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

#  define LAUNCH_KERNEL(T, DST_DTYPE)                                                                                  \
    do {                                                                                                               \
      dim3 grid(num_blocks);                                                                                           \
      dim3 block(num_threads);                                                                                         \
      if (is_column_major) {                                                                                           \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true><<<grid, block, 0, stream>>>(                             \
            static_cast<T*>(input), output_q, static_cast<float*>(output_s), group_size, num_groups, groups_per_block, \
            eps, min_8bit, max_8bit, /*scale_num_rows*/ n / group_size, /*scale_stride*/ m);                           \
      } else {                                                                                                         \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false>                                                         \
            <<<grid, block, 0, stream>>>(static_cast<T*>(input), output_q, static_cast<float*>(output_s), group_size,  \
                                         num_groups, groups_per_block, eps, min_8bit, max_8bit);                       \
      }                                                                                                                \
    } while (0)

  if (std::is_same<T, float>::value) {
    LAUNCH_KERNEL(float, __nv_fp8_e4m3);
  } else if (std::is_same<T, half>::value) {
    LAUNCH_KERNEL(half, __nv_fp8_e4m3);
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    LAUNCH_KERNEL(__nv_bfloat16, __nv_fp8_e4m3);
  }

#  undef LAUNCH_KERNEL
}

template <typename T>
void per_token_group_quant_fp8(void* input, void* output_q, void* output_s, int m, int n, int64_t group_size,
                                   bool is_column_major, cudaStream_t stream, float eps, float min_fp8, float max_fp8) {
  per_token_group_quant_8bit<T>(input, output_q, output_s, m, n, group_size, eps, min_fp8, max_fp8, is_column_major,
                                    stream);
}

#  define PER_TOKEN_GROUP_QUANT_FP8(T)                                                                            \
    template void per_token_group_quant_fp8<T>(void* input, void* output_q, void* output_s, int m, int n,     \
                                                   int64_t group_size, bool is_column_major, cudaStream_t stream, \
                                                   float eps, float min_fp8, float max_fp8)

PER_TOKEN_GROUP_QUANT_FP8(float);
PER_TOKEN_GROUP_QUANT_FP8(half);
#  ifdef ENABLE_BFLOAT16
PER_TOKEN_GROUP_QUANT_FP8(__nv_bfloat16);
#  endif
#  undef PER_TOKEN_GROUP_QUANT_FP8
#endif
}  // namespace nvidia
}  // namespace llm_kernels