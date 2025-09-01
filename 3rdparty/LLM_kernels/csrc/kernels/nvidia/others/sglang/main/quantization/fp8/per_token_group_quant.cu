/*
 * Adapted from
 * https://github.com/fzyzcjy/sglang/blob/feat/opt_quant_related/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu
 * https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/internode_ll.cu
 * https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh
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

#include "csrc/kernels/nvidia/others/sglang/main/quantization/fp8/per_token_group_quant.h"

#ifdef ENABLE_FP8
#  include <cuda_fp8.h>
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

#ifdef ENABLE_FP8

constexpr float LOCAL_ABSMAX_ABS = 1e-10;
constexpr uint32_t INPUT_PRIMARY_VEC_NUM_BYTES = 32;

template <int THREADS_PER_SUBWARP>
__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  constexpr unsigned mask = 0xffff;

  static_assert(
      (THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");

  if constexpr (THREADS_PER_SUBWARP >= 16) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  }
  if constexpr (THREADS_PER_SUBWARP >= 8) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  }
  if constexpr (THREADS_PER_SUBWARP >= 4) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  }
  if constexpr (THREADS_PER_SUBWARP >= 2) {
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  }
  return val;
}

__device__ __forceinline__ float tanh(const float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float silu(const float val) {
  // This is faster than `val / (1.0f + expf(-val))` but equivalent
  const float half = 0.5f * val;
  const float t = tanh(half);
  return half * (1.0f + t);
}

__device__ __forceinline__ void st_global(const int4* ptr, const int4& value) {
  asm volatile("st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
               "r"(value.w));
}

__device__ __forceinline__ int4 ld_global_nc(const int4* ptr) {
  int4 ret;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

template <typename T>
struct DtypeInfo;

template <>
struct DtypeInfo<__nv_fp8_e4m3> {
  static constexpr float MIN = -448;
  static constexpr float MAX = 448;
};

using scale_packed_t = float;
using scale_element_t = float;

template <typename T, typename DST_DTYPE, int GROUP_SIZE, int THREADS_PER_SUBWARP, bool IS_COLUMN_MAJOR = false,
          bool FUSE_SILU_MUL = false>
__device__ __forceinline__ void quant_fp8(const int thread_id, const int64_t block_group_id,
                                          const T* __restrict__ input, DST_DTYPE* __restrict__ output_q,
                                          scale_packed_t* __restrict__ output_s, const int hidden_size,
                                          const int subwarps_per_block, const int hidden_dim_num_groups,
                                          const int scale_hidden_stride) {
  using dst_dtype_info = DtypeInfo<DST_DTYPE>;

  const int64_t subwarp_id = thread_id / THREADS_PER_SUBWARP;
  const int lane_id = thread_id % THREADS_PER_SUBWARP;

  const int64_t group_id = block_group_id + subwarp_id;

  const int token_idx = group_id / hidden_dim_num_groups;
  // At the hidden_size dimension, we are handling idx-th group
  const int hidden_dim_group_idx = group_id % hidden_dim_num_groups;

  int64_t input_group_start_offset;
  if constexpr (FUSE_SILU_MUL) {
    // When fuse_silu_mul, the input shape is [token_num, hidden_size*2] instead of [token_num, hidden_size]
    input_group_start_offset = token_idx * hidden_size * 2 + hidden_dim_group_idx * GROUP_SIZE;
  } else {
    input_group_start_offset = group_id * GROUP_SIZE;
  }

  const int offset_num_groups = token_idx * hidden_dim_num_groups + hidden_dim_group_idx;

  constexpr uint32_t INPUT_PRIMARY_VEC_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(T);
  constexpr uint32_t INPUT_PRIMARY_INT4_SIZE = INPUT_PRIMARY_VEC_NUM_BYTES / sizeof(int4);

  int4 input_primary_int4[INPUT_PRIMARY_INT4_SIZE];
  T* input_primary_vec = reinterpret_cast<T*>(input_primary_int4);
  static_assert(sizeof(input_primary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_primary_int4));

  int4 input_secondary_int4[INPUT_PRIMARY_INT4_SIZE];
  T* input_secondary_vec = reinterpret_cast<T*>(input_secondary_int4);
  static_assert(sizeof(input_secondary_vec[0]) * INPUT_PRIMARY_VEC_SIZE == sizeof(input_secondary_int4));

  const T* input_offset = input + input_group_start_offset + lane_id * INPUT_PRIMARY_VEC_SIZE;
#  pragma unroll
  for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
    input_primary_int4[j] = ld_global_nc(reinterpret_cast<const int4*>(input_offset) + j);
  }

  if constexpr (FUSE_SILU_MUL) {
#  pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_INT4_SIZE; ++j) {
      input_secondary_int4[j] = ld_global_nc(reinterpret_cast<const int4*>(input_offset + hidden_size) + j);
    }
  }

  scale_element_t* scale_output;
  if constexpr (IS_COLUMN_MAJOR) {
    static_assert(sizeof(scale_packed_t) == sizeof(scale_element_t));

    constexpr int scale_token_stride = 1;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (hidden_dim_group_idx * scale_hidden_stride + token_idx * scale_token_stride);
  } else {
    scale_output = output_s + offset_num_groups;
  }

  float local_absmax = LOCAL_ABSMAX_ABS;

#  pragma unroll
  for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
    float val;
    if constexpr (FUSE_SILU_MUL) {
      const T val_lowprec =
          static_cast<T>(silu(static_cast<float>(input_primary_vec[j])) * static_cast<float>(input_secondary_vec[j]));
      val = static_cast<float>(val_lowprec);
      input_primary_vec[j] = val_lowprec;
    } else {
      val = static_cast<float>(input_primary_vec[j]);
    }

    const float abs_val = fabsf(val);
    local_absmax = fmaxf(local_absmax, abs_val);
  }

  local_absmax = GroupReduceMax<THREADS_PER_SUBWARP>(local_absmax, lane_id);

  // We use division instead of multiplication to maintain precision
  const float y_scale_inv = local_absmax / dst_dtype_info::MAX;

  if (lane_id == 0) {
    *scale_output = y_scale_inv;
  }

  int4 output_buf;
  static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE * sizeof(DST_DTYPE));

  if constexpr (std::is_same_v<DST_DTYPE, __nv_fp8_e4m3>) {
    const auto output_buf_ptr = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);
    static_assert(sizeof(output_buf) == INPUT_PRIMARY_VEC_SIZE / 2 * sizeof(__nv_fp8x2_storage_t));
    static_assert(INPUT_PRIMARY_VEC_SIZE % 2 == 0);

#  pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; j += 2) {
      float2 inputx2 = {static_cast<float>(input_primary_vec[j]), static_cast<float>(input_primary_vec[j + 1])};
      // We use division instead of multiplication to maintain precision
      float2 outputx2 = {inputx2.x / y_scale_inv, inputx2.y / y_scale_inv};
      output_buf_ptr[j / 2] = __nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3);
    }
  } else {
    const auto output_buf_ptr = reinterpret_cast<DST_DTYPE*>(&output_buf);

#  pragma unroll
    for (uint32_t j = 0; j < INPUT_PRIMARY_VEC_SIZE; ++j) {
      float val = static_cast<float>(input_primary_vec[j]);
      // We use division instead of multiplication to maintain precision
      float q_val = fminf(fmaxf(val / y_scale_inv, dst_dtype_info::MIN), dst_dtype_info::MAX);
      output_buf_ptr[j] = DST_DTYPE(q_val);
    }
  }

  st_global(reinterpret_cast<int4*>(output_q + offset_num_groups * GROUP_SIZE + lane_id * INPUT_PRIMARY_VEC_SIZE),
            output_buf);
}

template <typename T, typename DST_DTYPE, int GROUP_SIZE, int THREADS_PER_SUBWARP, bool IS_COLUMN_MAJOR = false,
          bool FUSE_SILU_MUL = false>
__global__ void per_token_group_quant_fp8_kernel(const T* __restrict__ input, DST_DTYPE* __restrict__ output_q,
                                                 scale_packed_t* __restrict__ output_s, const int hidden_size,
                                                 const int subwarps_per_block, const int hidden_dim_num_groups,
                                                 const int scale_hidden_stride = 0) {
  const int64_t block_group_id = blockIdx.x * subwarps_per_block;
  quant_fp8<T, DST_DTYPE, GROUP_SIZE, THREADS_PER_SUBWARP, IS_COLUMN_MAJOR, FUSE_SILU_MUL>(
      threadIdx.x, block_group_id, input, output_q, output_s, hidden_size, subwarps_per_block, hidden_dim_num_groups,
      scale_hidden_stride);
}

template <typename T, typename DST_DTYPE, int GROUP_SIZE, bool IS_COLUMN_MAJOR, bool FUSE_SILU_MUL>
void per_token_group_quant_fp8_kernel_launcher(const void* input, void* output_q, void* output_s, int m, int n,
                                               cudaStream_t stream) {
  constexpr int THREADS_PER_SUBWARP = GROUP_SIZE / 16;

  const int hidden_dim_num_groups = n / GROUP_SIZE;
  const int num_groups = m * hidden_dim_num_groups;
  const int subwarps_per_block = ([=]() -> int {
    if (num_groups % 16 == 0) {
      return 16;
    } else if (num_groups % 8 == 0) {
      return 8;
    } else if (num_groups % 4 == 0) {
      return 4;
    } else if (num_groups % 2 == 0) {
      return 2;
    }
    return 1;
  })();
  const int scale_hidden_stride = ([=]() -> int {
    if constexpr (IS_COLUMN_MAJOR) {
      return m;
    } else {
      return 0;
    }
  })();

  const int num_blocks = num_groups / subwarps_per_block;
  const int num_threads = subwarps_per_block * THREADS_PER_SUBWARP;
  dim3 grid(num_blocks);
  dim3 block(num_threads);
  const uint32_t smem_size = 0;
  per_token_group_quant_fp8_kernel<T, DST_DTYPE, GROUP_SIZE, THREADS_PER_SUBWARP, IS_COLUMN_MAJOR, FUSE_SILU_MUL>
      <<<grid, block, smem_size, stream>>>(static_cast<const T*>(input), static_cast<DST_DTYPE*>(output_q),
                                           static_cast<scale_packed_t*>(output_s), n, subwarps_per_block,
                                           hidden_dim_num_groups, scale_hidden_stride);
}

template <typename T>
void per_token_group_quant_fp8(const void* input, void* output_q, void* output_s, int m, int n, int64_t group_size,
                               bool is_column_major, bool fuse_silu_mul, cudaStream_t stream) {
#  define DISPATCH_FUSION(T, DST_DTYPE, GROUP_SIZE, IS_COLUMN_MAJOR)                                 \
    do {                                                                                             \
      if (fuse_silu_mul) {                                                                           \
        per_token_group_quant_fp8_kernel_launcher<T, DST_DTYPE, GROUP_SIZE, IS_COLUMN_MAJOR, true>(  \
            input, output_q, output_s, m, n, stream);                                                \
      } else {                                                                                       \
        per_token_group_quant_fp8_kernel_launcher<T, DST_DTYPE, GROUP_SIZE, IS_COLUMN_MAJOR, false>( \
            input, output_q, output_s, m, n, stream);                                                \
      }                                                                                              \
    } while (0)

#  define DISPATCH_COLUMN_MAJOR(T, DST_DTYPE, GROUP_SIZE) \
    do {                                                  \
      if (is_column_major) {                              \
        DISPATCH_FUSION(T, DST_DTYPE, GROUP_SIZE, true);  \
      } else {                                            \
        DISPATCH_FUSION(T, DST_DTYPE, GROUP_SIZE, false); \
      }                                                   \
    } while (0)

#  define DISPATCH_GROUP_SIZE(T, DST_DTYPE)            \
    do {                                               \
      switch (group_size) {                            \
        case 16:                                       \
          DISPATCH_COLUMN_MAJOR(T, DST_DTYPE, 16);     \
          break;                                       \
        case 32:                                       \
          DISPATCH_COLUMN_MAJOR(T, DST_DTYPE, 32);     \
          break;                                       \
        case 64:                                       \
          DISPATCH_COLUMN_MAJOR(T, DST_DTYPE, 64);     \
          break;                                       \
        case 128:                                      \
          DISPATCH_COLUMN_MAJOR(T, DST_DTYPE, 128);    \
          break;                                       \
        default:                                       \
          KLLM_KERNEL_THROW("Unsupported group_size"); \
      }                                                \
    } while (0)

  // dispatch data type
  if constexpr (std::is_same<T, half>::value) {
    DISPATCH_GROUP_SIZE(half, __nv_fp8_e4m3);
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    DISPATCH_GROUP_SIZE(__nv_bfloat16, __nv_fp8_e4m3);
  } else {
    KLLM_KERNEL_THROW("Unsupported data type");
  }

#  undef DISPATCH_GROUP_SIZE
#  undef DISPATCH_COLUMN_MAJOR
#  undef DISPATCH_FUSION
}

#  define PER_TOKEN_GROUP_QUANT_FP8(T)                                                                          \
    template void per_token_group_quant_fp8<T>(const void* input, void* output_q, void* output_s, int m, int n, \
                                               int64_t group_size, bool is_column_major, bool fuse_silu_mul,    \
                                               cudaStream_t stream);
PER_TOKEN_GROUP_QUANT_FP8(float);
PER_TOKEN_GROUP_QUANT_FP8(half);
PER_TOKEN_GROUP_QUANT_FP8(__nv_bfloat16);
#  undef PER_TOKEN_GROUP_QUANT_FP8

#endif  // ENABLE_FP8

}  // namespace nvidia
}  // namespace llm_kernels
