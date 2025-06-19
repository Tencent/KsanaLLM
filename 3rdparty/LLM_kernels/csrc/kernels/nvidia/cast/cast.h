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

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#if defined(ENABLE_FP8)
#  include <cuda_fp8.h>
#endif

namespace llm_kernels {
namespace nvidia {

// For float to [half, bfloat16]
void FloatToHalf(const float* input, size_t input_length, half* output, cudaStream_t stream);
void FloatToBFloat16(const float* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream);

// For half to [float, bfloat16]
void HalfToFloat(const half* input, size_t input_length, float* output, cudaStream_t stream,
                 const size_t input_stride = 1ul, const size_t output_stride = 1ul);
void HalfToBFloat16(void* data_ptr, size_t input_length, cudaStream_t stream);

// For bfloat16 to [float, half]
void BFloat16ToFloat(const __nv_bfloat16* input, size_t input_length, float* output, cudaStream_t stream,
                     const size_t input_stride = 1ul, const size_t output_stride = 1ul);
void BFloat16ToHalf(void* data_ptr, size_t input_length, cudaStream_t stream);

#if defined(ENABLE_FP8)
// For FP8E4M3 to [float, half, bfloat16]
void FloatToFp8E4M3(const float* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream);
void Fp8E4M3ToFloat(const __nv_fp8_e4m3* input, size_t input_length, float* output, cudaStream_t stream);
void HalfToFp8E4M3(const half* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream);
void Fp8E4M3ToHalf(const __nv_fp8_e4m3* input, size_t input_length, half* output, cudaStream_t stream);
void BFloat16ToFp8E4M3(const __nv_bfloat16* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream);
void Fp8E4M3ToBFloat16(const __nv_fp8_e4m3* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream);
#endif

// Int64 to int
void Int64ToInt(const int64_t* input, size_t input_length, int* output, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
