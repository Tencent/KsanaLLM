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

namespace llm_kernels {
namespace nvidia {

// Gaussian Error Linear Units: https://arxiv.org/abs/1606.08415
template <typename T>
struct GeluActivation;

// Rectified Linear Units: https://arxiv.org/abs/1803.08375
template <typename T>
struct ReluActivation;

// Sigmoid-Weighted Linear Units: https://arxiv.org/abs/1702.03118
template <typename T>
struct SiluActivation;

// Identity Activation = no activation
template <typename T>
struct IdentityActivation;

// for the tensor input with shape [m, n]
//  out = act(input[:, :n/2]) * input[:, n/2:]
template <template <typename T> class Activation, typename T>
void InvokeRowBasedActivation(T* out, const T* input, const int32_t m, const int32_t n, cudaStream_t& stream);

// fused act and matmul+bias into one kernel inplace
// for silu: data = act(data) * gated_weight
template <template <typename T> class Activation, typename T, typename BT>
void InvokeGenericActivation(T* out, const BT* bias, const T* gated_weights, const BT* gated_bias,
                             const int32_t* ia3_tasks, const T* ia3_weights, const int32_t m, const int32_t n,
                             const int32_t int8_mode, const float* activation_in, const float* activation_out,
                             const int32_t* padding_offset, const int32_t seq_len, cudaStream_t& stream);

template <template <typename T> class Activation, typename T, typename BT>
void InvokeGenericActivation(T* out, const BT* bias, const T* gated_weights, const BT* gated_bias,
                             const int32_t* ia3_tasks, const T* ia3_weights, const int32_t m, const int32_t n,
                             const int32_t int8_mode, const float* activation_in, const float* activation_out,
                             cudaStream_t& stream) {
  InvokeGenericActivation<Activation, T, BT>(out, bias, gated_weights, gated_bias, ia3_tasks, ia3_weights, m, n,
                                             int8_mode, activation_in, activation_out, (const int32_t*)nullptr, 0,
                                             stream);
}

template <typename T>
void InvokeAddBias(T* out, T const* bias, const int32_t m, const int32_t n, cudaStream_t& stream) {
  InvokeGenericActivation<IdentityActivation, T, T>(out, bias, nullptr, nullptr, nullptr, nullptr, m, n, 0, nullptr,
                                                    nullptr, stream);
}

template <typename T>
void InvokeAddBiasGeluV2(T* out, const T* bias, const int32_t* ia3_tasks, const T* ia3_weights,
                         const int32_t* padding_offset, const int32_t seq_len, const int32_t m, const int32_t n,
                         cudaStream_t& stream);

template <typename T>
void InvokeAddBiasGeluV2(T* out, const T* bias, const int32_t* ia3_tasks, const T* ia3_weights, const int32_t m,
                         const int32_t n, cudaStream_t& stream) {
  InvokeAddBiasGeluV2(out, bias, ia3_tasks, ia3_weights, nullptr, 0, m, n, stream);
}

template <typename T>
void InvokeAddBiasTanh(T* out, const T* bias, const int32_t m, const int32_t n, cudaStream_t& stream);

template <typename T>
void InvokeSigmoid(T* data, const int32_t size, const float scale, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
