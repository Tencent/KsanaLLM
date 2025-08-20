/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
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

namespace llm_kernels {
namespace nvidia {

// Fused Add-Multiply operations to replace scale_depth usage in add kernels

/**
 * @brief Compute output = (input1 + input2) * scale
 * @param output Output tensor
 * @param input1 First input tensor
 * @param input2 Second input tensor  
 * @param scale Scale factor to multiply
 * @param m Number of rows
 * @param n Number of columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeAddThenMul(T* output, const T* input1, const T* input2, const T scale, 
                      const int32_t m, const int32_t n, cudaStream_t stream);

/**
 * @brief Compute output = input1 + input2 * scale
 * @param output Output tensor
 * @param input1 First input tensor (not scaled)
 * @param input2 Second input tensor (to be scaled)
 * @param scale Scale factor for input2
 * @param m Number of rows
 * @param n Number of columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeAddMulSecond(T* output, const T* input1, const T* input2, const T scale,
                        const int32_t m, const int32_t n, cudaStream_t stream);

/**
 * @brief Compute output = (input1 + input2 + bias) * scale
 * @param output Output tensor
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param bias Bias tensor (broadcasted along columns)
 * @param scale Scale factor to multiply
 * @param m Number of rows
 * @param n Number of columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeAddBiasThenMul(T* output, const T* input1, const T* input2, const T* bias, const T scale,
                          const int32_t m, const int32_t n, cudaStream_t stream);

/**
 * @brief Compute output = input1 * scale1 + input2 * scale2
 * @param output Output tensor
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param scale1 Scale factor for input1
 * @param scale2 Scale factor for input2
 * @param m Number of rows
 * @param n Number of columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeMulThenAdd(T* output, const T* input1, const T* input2, const T scale1, const T scale2,
                      const int32_t m, const int32_t n, cudaStream_t stream);

/**
 * @brief Compute output = (input1 + residual1 + residual2 + bias) * scale
 * @param output Output tensor
 * @param input1 Main input tensor
 * @param residual1 First residual connection (can be nullptr)
 * @param residual2 Second residual connection (can be nullptr)
 * @param bias Bias tensor (can be nullptr)
 * @param scale Scale factor to multiply
 * @param m Number of rows
 * @param n Number of columns
 * @param stream CUDA stream
 */
template <typename T>
void InvokeAddResidualsBiasThenMul(T* output, const T* input1, const T* residual1, const T* residual2, 
                                   const T* bias, const T scale, const int32_t m, const int32_t n, 
                                   cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels 