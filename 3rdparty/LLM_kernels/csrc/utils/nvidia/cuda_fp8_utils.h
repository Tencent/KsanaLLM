/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#ifdef ENABLE_FP8
#  include <stdint.h>
#  include <stdio.h>

#  include <algorithm>

#  include <cuda_fp8.h>
#  include <cuda_runtime.h>

// #define FP8_MHA
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
#    define FUSE_GEMM_ACT
#  endif
#  define FP8_GEMM_OUTPUT_QUANT_DISABLE

#  ifdef FUSE_GEMM_ACT
#    define USE_QGMMA
#  endif

namespace llm_kernels {
namespace utils {

constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E4M3_MIN = -448.0f;
constexpr float FP8_E4M3_MIN_SCALE = 1.0f / FP8_E4M3_MAX / 512.0f;
constexpr size_t VEC_MEM_ACC_SIZE = 16;  // Size in bytes used for vectorized memory accesses

// Packed Data Type
typedef struct __CUDA_ALIGN__(32) {
  float array[8];
} float8;

typedef struct __CUDA_ALIGN__(16) {
  half array[8];
} half8;

#  ifdef ENABLE_FP8
typedef struct __CUDA_ALIGN__(2) {
  __nv_fp8_e4m3 array[2];
} __nv_fp8_2_e4m3;

typedef struct __CUDA_ALIGN__(4) {
  __nv_fp8_e4m3 array[4];
} __nv_fp8_4_e4m3;

typedef struct __CUDA_ALIGN__(4) {
  __nv_fp8x2_e4m3 array[2];
} __nv_fp8x2_x2_e4m3;

typedef struct __CUDA_ALIGN__(8) {
  __nv_fp8_e4m3 array[8];
} __nv_fp8_8_e4m3;

typedef struct __CUDA_ALIGN__(8) {
  __nv_fp8x2_e4m3 array[4];
} __nv_fp8x2_x4_e4m3;

typedef struct __CUDA_ALIGN__(16) {
  __nv_fp8_e4m3 array[16];
} __nv_fp8x16_e4m3;
#  endif

// NOTE(karlluo): for override and fallback cuda default function, we need all this function name with cuda's code
// format rule
__inline__ __device__ void fp8x4_e4m3_to_bfloat2(__nv_bfloat162* out1, __nv_bfloat162* out2,
                                                 const __nv_fp8x4_e4m3* in) {
  const char4 tmp_val = reinterpret_cast<const char4*>(in)[0];
  *out1 = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                         (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
  *out2 = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.z)[0],
                         (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.w)[0]);
}

__inline__ __device__ __nv_bfloat162 fp8x2_e4m3_to_bfloat2(const __nv_fp8x2_e4m3* in) {
  const char2 tmp_val = reinterpret_cast<const char2*>(in)[0];
  __nv_bfloat162 out = __nv_bfloat162((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                                      (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
  return out;
}

__inline__ __device__ void fp8x4_e4m3_to_half2(half2* out1, half2* out2, const __nv_fp8x4_e4m3* in) {
  const char4 tmp_val = reinterpret_cast<const char4*>(in)[0];
  *out1 = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
  *out2 = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.z)[0],
                (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.w)[0]);
}

__inline__ __device__ half2 fp8x2_e4m3_to_half2(const __nv_fp8x2_e4m3* in) {
  const char2 tmp_val = reinterpret_cast<const char2*>(in)[0];
  half2 out = half2((float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.x)[0],
                    (float)reinterpret_cast<const __nv_fp8_e4m3*>(&tmp_val.y)[0]);
  return out;
}

template <typename T_OUT, typename T_IN>
void InvokeQuantizeMatrix(T_OUT* output, float const* scale, T_IN const* input, size_t num_channels,
                          size_t channel_size, cudaStream_t stream);

template <typename T_OUT, typename T_IN, typename T_FAKE>
void InvokeFakeQuantize(T_OUT* dst, const T_IN* src, const int32_t size, cudaStream_t stream);

template <typename T_IN>
void InvokeComputeFP8QuantizeScale(float* output, const T_IN* input, const int32_t num_channels,
                                   const int32_t channel_size, cudaStream_t stream);

void InvokeRescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                          cudaStream_t& stream);
}  // namespace utils
}  // namespace llm_kernels
#endif  // ENABLE_FP8
