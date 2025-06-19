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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "csrc/kernels/nvidia/gptq_marlin/marlin.cuh"

namespace llm_kernels {
namespace nvidia {
namespace marlin {

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<half> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) { return __half2float(x); }

  static __device__ half2 inline num2num2(const half x) { return __half2half2(x); }

  static __device__ half2 inline nums2num2(const half x1, const half x2) { return __halves2half2(x1, x2); }

  static __host__ __device__ half inline float2num(const float x) { return __float2half(x); }
};

template <>
class ScalarType<nv_bfloat16> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

  static __device__ float inline num2float(const nv_bfloat16 x) { return __bfloat162float(x); }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) { return __bfloat162bfloat162(x); }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1, const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) { return __float2bfloat16(x); }
};

}  // namespace marlin
}  // namespace nvidia
}  // namespace llm_kernels