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

#include "csrc/kernels/nvidia/add_mul/add_mul.h"

#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

constexpr int32_t ADD_MUL_BLOCK_SIZE = 256;

/**
 * @brief CUDA kernel for computing output = (input1 + input2) * scale
 */
template <typename T>
__global__ void AddThenMulKernel(T* output, const T* __restrict__ input1, const T* __restrict__ input2,
                                 const T scale, const int32_t total_element_num) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    T sum = input1[index] + input2[index];
    output[index] = sum * scale;
  }
}

/**
 * @brief Vectorized CUDA kernel for computing output = (input1 + input2) * scale
 */
template <typename VecT, typename ScalarT>
__global__ void AddThenMulVecKernel(VecT* output, const VecT* __restrict__ input1, const VecT* __restrict__ input2,
                                    const ScalarT scale, const int32_t total_element_num) {
  const int32_t index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < total_element_num) {
    const int32_t idx = index / 4;
    VecT input1_val = input1[idx];
    VecT input2_val = input2[idx];
    VecT output_val;

    // Vectorized computation for better performance
    output_val.x = (input1_val.x + input2_val.x) * scale;
    output_val.y = (input1_val.y + input2_val.y) * scale;
    output_val.z = (input1_val.z + input2_val.z) * scale;
    output_val.w = (input1_val.w + input2_val.w) * scale;
    
    output[idx] = output_val;
  }
}

/**
 * @brief CUDA kernel for computing output = input1 + input2 * scale
 */
template <typename T>
__global__ void AddMulSecondKernel(T* output, const T* __restrict__ input1, const T* __restrict__ input2,
                                   const T scale, const int32_t total_element_num) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    output[index] = input1[index] + input2[index] * scale;
  }
}

/**
 * @brief CUDA kernel for computing output = (input1 + input2 + bias) * scale
 */
template <typename T>
__global__ void AddBiasThenMulKernel(T* output, const T* __restrict__ input1, const T* __restrict__ input2,
                                     const T* __restrict__ bias, const T scale, const int32_t total_element_num,
                                     const int32_t n) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    T bias_val = (bias == nullptr) ? static_cast<T>(0.0f) : bias[index % n];
    T sum = input1[index] + input2[index] + bias_val;
    output[index] = sum * scale;
  }
}

/**
 * @brief CUDA kernel for computing output = input1 * scale1 + input2 * scale2
 */
template <typename T>
__global__ void MulThenAddKernel(T* output, const T* __restrict__ input1, const T* __restrict__ input2,
                                 const T scale1, const T scale2, const int32_t total_element_num) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    output[index] = input1[index] * scale1 + input2[index] * scale2;
  }
}

/**
 * @brief CUDA kernel for computing output = (input1 + residual1 + residual2 + bias) * scale
 */
template <typename T>
__global__ void AddResidualsBiasThenMulKernel(T* output, const T* __restrict__ input1,
                                              const T* __restrict__ residual1, const T* __restrict__ residual2,
                                              const T* __restrict__ bias, const T scale,
                                              const int32_t total_element_num, const int32_t n) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    T bias_val = (bias == nullptr) ? static_cast<T>(0.0f) : bias[index % n];
    T residual1_val = (residual1 == nullptr) ? static_cast<T>(0.0f) : residual1[index];
    T residual2_val = (residual2 == nullptr) ? static_cast<T>(0.0f) : residual2[index];
    
    T sum = input1[index] + residual1_val + residual2_val + bias_val;
    output[index] = sum * scale;
  }
}

// Implementation functions

template <typename T>
void InvokeAddThenMul(T* output, const T* input1, const T* input2, const T scale, 
                      const int32_t m, const int32_t n, cudaStream_t stream) {
  const int32_t total_element_num = m * n;
  const size_t kVecSize = 4;
  
  // Use vectorized kernel when possible for better performance
  if (total_element_num % kVecSize == 0) {
    const int32_t block_num = ceil(float(total_element_num) / (ADD_MUL_BLOCK_SIZE * kVecSize));
    dim3 grid(block_num);
    dim3 block(ADD_MUL_BLOCK_SIZE);
    using VecType = typename utils::PackType<T, kVecSize>::type;
    AddThenMulVecKernel<VecType, T><<<grid, block, 0, stream>>>(
        reinterpret_cast<VecType*>(output), 
        reinterpret_cast<const VecType*>(input1),
        reinterpret_cast<const VecType*>(input2),
        scale, total_element_num);
  } else {
    const int32_t block_num = ceil(float(total_element_num) / ADD_MUL_BLOCK_SIZE);
    dim3 grid(block_num);
    dim3 block(ADD_MUL_BLOCK_SIZE);
    AddThenMulKernel<T><<<grid, block, 0, stream>>>(output, input1, input2, scale, total_element_num);
  }
}

template <typename T>
void InvokeAddMulSecond(T* output, const T* input1, const T* input2, const T scale,
                        const int32_t m, const int32_t n, cudaStream_t stream) {
  const int32_t total_element_num = m * n;
  const int32_t block_num = ceil(float(total_element_num) / ADD_MUL_BLOCK_SIZE);
  dim3 grid(block_num);
  dim3 block(ADD_MUL_BLOCK_SIZE);
  AddMulSecondKernel<T><<<grid, block, 0, stream>>>(output, input1, input2, scale, total_element_num);
}

template <typename T>
void InvokeAddBiasThenMul(T* output, const T* input1, const T* input2, const T* bias, const T scale,
                          const int32_t m, const int32_t n, cudaStream_t stream) {
  const int32_t total_element_num = m * n;
  const int32_t block_num = ceil(float(total_element_num) / ADD_MUL_BLOCK_SIZE);
  dim3 grid(block_num);
  dim3 block(ADD_MUL_BLOCK_SIZE);
  AddBiasThenMulKernel<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, scale, total_element_num, n);
}

template <typename T>
void InvokeMulThenAdd(T* output, const T* input1, const T* input2, const T scale1, const T scale2,
                      const int32_t m, const int32_t n, cudaStream_t stream) {
  const int32_t total_element_num = m * n;
  const int32_t block_num = ceil(float(total_element_num) / ADD_MUL_BLOCK_SIZE);
  dim3 grid(block_num);
  dim3 block(ADD_MUL_BLOCK_SIZE);
  MulThenAddKernel<T><<<grid, block, 0, stream>>>(output, input1, input2, scale1, scale2, total_element_num);
}

template <typename T>
void InvokeAddResidualsBiasThenMul(T* output, const T* input1, const T* residual1, const T* residual2, 
                                   const T* bias, const T scale, const int32_t m, const int32_t n, 
                                   cudaStream_t stream) {
  const int32_t total_element_num = m * n;
  const int32_t block_num = ceil(float(total_element_num) / ADD_MUL_BLOCK_SIZE);
  dim3 grid(block_num);
  dim3 block(ADD_MUL_BLOCK_SIZE);
  AddResidualsBiasThenMulKernel<T><<<grid, block, 0, stream>>>(
      output, input1, residual1, residual2, bias, scale, total_element_num, n);
}

// Explicit template instantiations for supported types

#define INSTANTIATE_ADD_THEN_MUL(T) \
  template void InvokeAddThenMul<T>(T* output, const T* input1, const T* input2, const T scale, \
                                    const int32_t m, const int32_t n, cudaStream_t stream)

INSTANTIATE_ADD_THEN_MUL(float);
INSTANTIATE_ADD_THEN_MUL(half);
INSTANTIATE_ADD_THEN_MUL(__nv_bfloat16);
#undef INSTANTIATE_ADD_THEN_MUL

#define INSTANTIATE_ADD_MUL_SECOND(T) \
  template void InvokeAddMulSecond<T>(T* output, const T* input1, const T* input2, const T scale, \
                                      const int32_t m, const int32_t n, cudaStream_t stream)

INSTANTIATE_ADD_MUL_SECOND(float);
INSTANTIATE_ADD_MUL_SECOND(half);
INSTANTIATE_ADD_MUL_SECOND(__nv_bfloat16);
#undef INSTANTIATE_ADD_MUL_SECOND

#define INSTANTIATE_ADD_BIAS_THEN_MUL(T) \
  template void InvokeAddBiasThenMul<T>(T* output, const T* input1, const T* input2, const T* bias, \
                                        const T scale, const int32_t m, const int32_t n, cudaStream_t stream)

INSTANTIATE_ADD_BIAS_THEN_MUL(float);
INSTANTIATE_ADD_BIAS_THEN_MUL(half);
INSTANTIATE_ADD_BIAS_THEN_MUL(__nv_bfloat16);
#undef INSTANTIATE_ADD_BIAS_THEN_MUL

#define INSTANTIATE_MUL_THEN_ADD(T) \
  template void InvokeMulThenAdd<T>(T* output, const T* input1, const T* input2, const T scale1, \
                                    const T scale2, const int32_t m, const int32_t n, cudaStream_t stream)

INSTANTIATE_MUL_THEN_ADD(float);
INSTANTIATE_MUL_THEN_ADD(half);
INSTANTIATE_MUL_THEN_ADD(__nv_bfloat16);
#undef INSTANTIATE_MUL_THEN_ADD

#define INSTANTIATE_ADD_RESIDUALS_BIAS_THEN_MUL(T) \
  template void InvokeAddResidualsBiasThenMul<T>(T* output, const T* input1, const T* residual1, \
                                                 const T* residual2, const T* bias, const T scale, \
                                                 const int32_t m, const int32_t n, cudaStream_t stream)

INSTANTIATE_ADD_RESIDUALS_BIAS_THEN_MUL(float);
INSTANTIATE_ADD_RESIDUALS_BIAS_THEN_MUL(half);
INSTANTIATE_ADD_RESIDUALS_BIAS_THEN_MUL(__nv_bfloat16);
#undef INSTANTIATE_ADD_RESIDUALS_BIAS_THEN_MUL

}  // namespace nvidia
}  // namespace llm_kernels 