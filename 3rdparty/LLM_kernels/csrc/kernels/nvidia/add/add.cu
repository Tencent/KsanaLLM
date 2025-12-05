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

#include "csrc/kernels/nvidia/add/add.h"

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

constexpr int32_t ADD_BIAS_RES_BLOCK_SIZE = 1024;

template <typename T, int32_t RESIDUAL_NUM, typename T2 = T>
__global__ void AddBiasResidualKernel(T* output, const T2* __restrict__ input, const T* __restrict__ residual1,
                                      const T* __restrict__ residual2, const T* __restrict__ bias,
                                      const float* __restrict__ scale_inter, const float* __restrict__ scale_out,
                                      const int32_t total_element_num, const int32_t n) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_element_num) {
    T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[index % n];
    T in;
    if (std::is_same<T, T2>::value) {
      in = CastCudaDataType<T>(input[index]);  // cast required for compilation when T != T2
    } else {
      in = CastCudaDataType<float>(input[index]) * (*scale_inter) * (*scale_out);
    }

    if (RESIDUAL_NUM == 1) {
      if (residual1) {
        output[index] = in + residual1[index] + bias_val;
      } else {
        output[index] = in + bias_val;
      }
    } else if (RESIDUAL_NUM == 2) {
      output[index] = in + residual1[index] + residual2[index] + bias_val;
    }
  }
}

template <typename T, int kVecSize>
__global__ void AddResidualKernel(T* __restrict__ output, const T* __restrict__ input, const T* __restrict__ residual,
                                  const int32_t total_element_num) {
  if (const int32_t index = kVecSize * (blockIdx.x * blockDim.x + threadIdx.x); index < total_element_num) {
    vec_t<T, kVecSize> input_vec;
    input_vec.load(input + index);
    vec_t<T, kVecSize> residual_vec;
    residual_vec.load(residual + index);
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      input_vec[i] += residual_vec[i];
    }
    input_vec.store(output + index);
  }
}

template <typename T>
void InvokeAddBiasResidual(T* output, const T* input, const T* residual1, const T* residual2, const T* bias,
                           const float* scale_inter, const float* scale_out, const int32_t m, const int32_t n,
                           cudaStream_t stream) {
  if (((scale_inter == nullptr) ^ (scale_out == nullptr))) {
    throw std::runtime_error("Cannot use `scale_inter` without `scale_out`");
  }
  constexpr int32_t BLOCK_SIZE = 256;
  int32_t total_element_num = m * n;
  const bool should_scale_input = scale_inter != nullptr;
  if (should_scale_input) {
    int32_t block_num = ceil(float(m * n) / BLOCK_SIZE);
    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);
    if (residual2 == nullptr) {
      AddBiasResidualKernel<T, 1><<<grid, block, 0, stream>>>(output, reinterpret_cast<const int32_t*>(input),
                                                              residual1, residual2, bias, scale_inter, scale_out,
                                                              total_element_num, n);
    } else {
      AddBiasResidualKernel<T, 2><<<grid, block, 0, stream>>>(output, reinterpret_cast<const int32_t*>(input),
                                                              residual1, residual2, bias, scale_inter, scale_out,
                                                              total_element_num, n);
    }
  } else {
    constexpr size_t kVecSize = sizeof(int4) / sizeof(T);
    if (bias == nullptr && residual2 == nullptr && (total_element_num & (kVecSize - 1)) == 0) {
      const uint32_t block_num = (total_element_num + kVecSize * BLOCK_SIZE - 1) / (kVecSize * BLOCK_SIZE);
      dim3 grid(block_num);
      dim3 block(BLOCK_SIZE);
      AddResidualKernel<T, kVecSize><<<grid, block, 0, stream>>>(output, input, residual1, total_element_num);
    } else {
      const uint32_t block_num = (total_element_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
      dim3 grid(block_num);
      dim3 block(BLOCK_SIZE);
      if (residual2 == nullptr) {
        AddBiasResidualKernel<T, 1><<<grid, block, 0, stream>>>(output, input, residual1, residual2, bias, nullptr,
                                                                nullptr, total_element_num, n);
      } else {
        AddBiasResidualKernel<T, 2><<<grid, block, 0, stream>>>(output, input, residual1, residual2, bias, nullptr,
                                                                nullptr, total_element_num, n);
      }
    }
  }
}

template <typename T>
void InvokeAddBiasResidual(T* output, const T* residual1, const T* residual2, const T* bias, const int32_t m,
                           const int32_t n, cudaStream_t stream) {
  InvokeAddBiasResidual(output, output, residual1, residual2, bias, nullptr, nullptr, m, n, stream);
}

template <typename T>
__global__ void AddBiasAttentionFfnResidualKernel(T* block_output, const T* __restrict__ ffn_output,
                                                  const T* __restrict__ attn_output, const T* __restrict__ block_input,
                                                  const T* __restrict__ bias, const int32_t m, const int32_t n,
                                                  const int32_t block_input_tp_split) {
  const int32_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
  if (col_index < n) {
    block_output[blockIdx.x * n + col_index] =
        ffn_output[blockIdx.x * n + col_index] + attn_output[blockIdx.x * n + col_index] + bias[col_index] +
        ((block_input != nullptr)
             ? CastCudaDataType<T>((float)block_input[blockIdx.x * n + col_index] / (float)block_input_tp_split)
             : static_cast<T>(0.0f));
  }
}

template <typename T>
__global__ void AddBiasAttentionFfnResidualKernel(T* block_output, const T* __restrict__ ffn_output,
                                                  const T* __restrict__ attn_output, const T* __restrict__ bias,
                                                  const int32_t m, const int32_t n,
                                                  const int32_t block_input_tp_split) {
  const int32_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
  if (col_index < n) {
    const int32_t global_index = blockIdx.x * n + col_index;
    block_output[global_index] =
        add(CastCudaDataType<T>((float)block_output[global_index] / (float)block_input_tp_split),
            ffn_output[global_index], attn_output[global_index], bias[col_index]);
  }
}

template <typename T>
void InvokeAddBiasAttentionFfnResidual(T* block_output, const T* ffn_output, const T* attn_output, const T* block_input,
                                       const T* bias, const int32_t m, const int32_t n,
                                       const int32_t block_input_tp_split, cudaStream_t stream) {
  int32_t blocks_per_row = ceil(float(n) / ADD_BIAS_RES_BLOCK_SIZE);
  dim3 grid(m, blocks_per_row);
  dim3 block(min(n, ADD_BIAS_RES_BLOCK_SIZE));
  if (block_output == block_input) {
    AddBiasAttentionFfnResidualKernel<<<grid, block, 0, stream>>>(block_output, ffn_output, attn_output, bias, m, n,
                                                                  block_input_tp_split);
  } else {
    AddBiasAttentionFfnResidualKernel<<<grid, block, 0, stream>>>(block_output, ffn_output, attn_output, block_input,
                                                                  bias, m, n, block_input_tp_split);
  }
}

#define INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(T)                                                          \
  template void InvokeAddBiasResidual(T* output, const T* input, const T* residual1, const T* residual2, \
                                      const T* bias, const float* scale_inter, const float* scale_out,   \
                                      const int32_t m, const int32_t n, cudaStream_t stream)
INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(float);
INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(half);
INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL(__nv_bfloat16);
#undef INSTANTIATE_INVOKE_ADD_BIAS_RESIDUAL

template void InvokeAddBiasResidual(float* output, const float* residual1, const float* residual2, const float* bias,
                                    const int32_t m, const int32_t n, cudaStream_t stream);

template void InvokeAddBiasResidual(half* output, const half* residual1, const half* residual2, const half* bias,
                                    const int32_t m, const int32_t n, cudaStream_t stream);
template void InvokeAddBiasResidual(__nv_bfloat16* output, const __nv_bfloat16* residual1,
                                    const __nv_bfloat16* residual2, const __nv_bfloat16* bias, const int32_t m,
                                    const int32_t n, cudaStream_t stream);

template void InvokeAddBiasAttentionFfnResidual(float* block_output, const float* ffn_output, const float* attn_output,
                                                const float* input, const float* bias, const int32_t m, const int32_t n,
                                                const int32_t block_input_tp_split, cudaStream_t stream);

template void InvokeAddBiasAttentionFfnResidual(half* block_output, const half* ffn_output, const half* attn_output,
                                                const half* input, const half* bias, const int32_t m, const int32_t n,
                                                const int32_t block_input_tp_split, cudaStream_t stream);
template void InvokeAddBiasAttentionFfnResidual(__nv_bfloat16* block_output, const __nv_bfloat16* ffn_output,
                                                const __nv_bfloat16* attn_output, const __nv_bfloat16* input,
                                                const __nv_bfloat16* bias, const int32_t m, const int32_t n,
                                                const int32_t block_input_tp_split, cudaStream_t stream);

// NOTE(karlluo): inplace case output can't use __restrict__
template <typename T>
__global__ void InvokeT5addResidualKernel(T* output, const T* __restrict__ input, const int32_t m, const int32_t n) {
  const int32_t col_index = blockIdx.y * blockDim.x + threadIdx.x;
  if (col_index < n) {
    float out_val = (float)output[blockIdx.x * n + col_index] + (float)input[blockIdx.x * n + col_index];
    output[blockIdx.x * n + col_index] =
        (T)((std::is_same<T, half>::value && (out_val > 64512 || out_val < -64512)) ? (out_val > 0 ? 64512 : -64512)
                                                                                    : out_val);
  }
}

template <typename T>
void InvokeT5AddResidual(T* output, const T* input, const int32_t m, const int32_t n, cudaStream_t stream) {
  int32_t blocks_per_row = ceil(float(n) / ADD_BIAS_RES_BLOCK_SIZE);
  dim3 grid(m, blocks_per_row);
  dim3 block(min(n, ADD_BIAS_RES_BLOCK_SIZE));
  InvokeT5addResidualKernel<<<grid, block, 0, stream>>>(output, input, m, n);
}

template void InvokeT5AddResidual(float* output, const float* input, const int32_t m, const int32_t n,
                                  cudaStream_t stream);
template void InvokeT5AddResidual(half* output, const half* input, const int32_t m, const int32_t n,
                                  cudaStream_t stream);
template void InvokeT5AddResidual(__nv_bfloat16* output, const __nv_bfloat16* input, const int32_t m, const int32_t n,
                                  cudaStream_t stream);

template <typename T>
void InvokeT5AddBiasResidual(T* output, const T* input, const T* bias, const int32_t m, const int32_t n,
                             cudaStream_t stream) {
  if (bias != nullptr) {
    InvokeAddBiasResidual(output, input, bias, m, n, stream);
  } else {
    InvokeT5AddResidual(output, input, m, n, stream);
  }
  return;
}

template void InvokeT5AddBiasResidual(float* output, const float* input, const float* bias, const int32_t m,
                                      const int32_t n, cudaStream_t stream);
template void InvokeT5AddBiasResidual(half* output, const half* input, const half* bias, const int32_t m,
                                      const int32_t n, cudaStream_t stream);
template void InvokeT5AddBiasResidual(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* bias,
                                      const int32_t m, const int32_t n, cudaStream_t stream);

// InvokeAddBiasResidualCol32 input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n) (grid, block)
// must be (m, n/4) using char4
template <typename T>
__global__ void AddBiasInputCOL32Int8IDataTypeOKernel(T* output, const int8_t* __restrict__ input1,
                                                      const T* __restrict__ input2, const T* __restrict__ bias,
                                                      int32_t m, int32_t n,
                                                      const float* __restrict__ input1_deq_factor_ptr) {
  const float input1_deQFactor = __ldg(input1_deq_factor_ptr);
  int32_t col_start = threadIdx.x << 2;

  float local_out[4];
  int32_t outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
  char4* input1TmpPtr = (char4*)input1;
  char4 input1Tmp = __ldg(input1TmpPtr + outIdx);

  int32_t col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2[(outIdx << 2) + 0]) + static_cast<float>(input1Tmp.x) * input1_deQFactor +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2[(outIdx << 2) + 1]) + static_cast<float>(input1Tmp.y) * input1_deQFactor +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2[(outIdx << 2) + 2]) + static_cast<float>(input1Tmp.z) * input1_deQFactor +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2[(outIdx << 2) + 3]) + static_cast<float>(input1Tmp.w) * input1_deQFactor +
                 static_cast<float>(__ldg(bias + col_start_tmp));

  for (int32_t i = 0; i < 4; i++) {
    output[(outIdx << 2) + i] = static_cast<T>(local_out[i]);
  }
}

template <>
__global__ void AddBiasInputCOL32Int8IDataTypeOKernel(half4* output, const int8_t* __restrict__ input1,
                                                      const half4* __restrict__ input2, const half4* __restrict__ bias,
                                                      int32_t m, int32_t n,
                                                      const float* __restrict__ input1_deq_factor_ptr) {
  const float input1_deQFactor = __ldg(input1_deq_factor_ptr);
  int32_t col_start = (blockIdx.x << 5) + (threadIdx.x << 2);
  int32_t row_start = (blockIdx.y << 5) + (threadIdx.y);

  if (col_start < n && row_start < m) {
    half4 local_out;
    int32_t outIdx = ((col_start & 0xffffffe0) * m + (row_start << 5) + (col_start & 31)) >> 2;
    char4* input1TmpPtr = (char4*)input1;
    char4 input1Tmp = input1TmpPtr[outIdx];
    half4 input2Tmp = input2[outIdx];
    half4 biasTmp = bias[col_start >> 2];

    local_out.x = static_cast<half>((float)input1Tmp.x * input1_deQFactor + (float)biasTmp.x + (float)input2Tmp.x);
    local_out.y = static_cast<half>((float)input1Tmp.y * input1_deQFactor + (float)biasTmp.y + (float)input2Tmp.y);
    local_out.z = static_cast<half>((float)input1Tmp.z * input1_deQFactor + (float)biasTmp.z + (float)input2Tmp.z);
    local_out.w = static_cast<half>((float)input1Tmp.w * input1_deQFactor + (float)biasTmp.w + (float)input2Tmp.w);
    output[outIdx] = local_out;
  }
}

template <typename T>
void InvokeAddBiasResidualCol32(T* output, const int8_t* input1, const T* input2, const T* bias, int32_t m, int32_t n,
                                cudaStream_t stream, const float* input1_deq_factor_ptr) {
  dim3 grid((n + DEFAULT_CUDA_WARP_SIZE - 1) / DEFAULT_CUDA_WARP_SIZE,
            (m + DEFAULT_CUDA_WARP_SIZE - 1) / DEFAULT_CUDA_WARP_SIZE);
  dim3 block(DEFAULT_CUDA_QUARTER_WARP_SIZE, DEFAULT_CUDA_WARP_SIZE);

  if (sizeof(T) == 2) {
    AddBiasInputCOL32Int8IDataTypeOKernel<<<grid, block, 0, stream>>>((half4*)output, input1, (const half4*)input2,
                                                                      (const half4*)bias, m, n, input1_deq_factor_ptr);
  } else {
    AddBiasInputCOL32Int8IDataTypeOKernel<T>
        <<<grid, block, 0, stream>>>(output, input1, input2, bias, m, n, input1_deq_factor_ptr);
  }
}

template void InvokeAddBiasResidualCol32(float* output, const int8_t* input1, const float* input2, const float* bias,
                                         int32_t m, int32_t n, cudaStream_t stream, const float* input1_deq_factor_ptr);

template void InvokeAddBiasResidualCol32(half* output, const int8_t* input1, const half* input2, const half* bias,
                                         int32_t m, int32_t n, cudaStream_t stream, const float* input1_deq_factor_ptr);

// InvokeAddBiasResidualCol32 input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n) (grid, block)
// must be (m, n/4) using char4
template <typename T>
__global__ void AddBiasInputCOL32Int32IDataTypeOKernel(T* output, const int32_t* __restrict__ input1,
                                                       const T* __restrict__ input2, const T* __restrict__ bias,
                                                       int32_t m, int32_t n, const float* __restrict__ weight_amax,
                                                       const float* __restrict__ input1_amax_ptr,
                                                       const int32_t scale_is_vector) {
  int32_t col_start = threadIdx.x << 2;
  const float4* weight_scale_ptr = (const float4*)weight_amax;
  const float4 weight_scale = __ldg(weight_scale_ptr + threadIdx.x * scale_is_vector);
  const float input1_deQ = __ldg(input1_amax_ptr) / 127.0f;

  float local_out[4];
  int32_t outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
  int4* input1TmpPtr = (int4*)input1;
  int4 input1Tmp = input1TmpPtr[outIdx];

  int32_t col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2[(outIdx << 2) + 0]) +
                 static_cast<float>(input1Tmp.x) * input1_deQ * weight_scale.x / 127.0f +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2[(outIdx << 2) + 1]) +
                 static_cast<float>(input1Tmp.y) * input1_deQ * weight_scale.y / 127.0f +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2[(outIdx << 2) + 2]) +
                 static_cast<float>(input1Tmp.z) * input1_deQ * weight_scale.z / 127.0f +
                 static_cast<float>(__ldg(bias + col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2[(outIdx << 2) + 3]) +
                 static_cast<float>(input1Tmp.w) * input1_deQ * weight_scale.w / 127.0f +
                 static_cast<float>(__ldg(bias + col_start_tmp));

  for (int32_t i = 0; i < 4; i++) {
    output[(outIdx << 2) + i] = static_cast<T>(local_out[i]);
  }
}

template <>
__global__ void AddBiasInputCOL32Int32IDataTypeOKernel(half4* output, const int32_t* __restrict__ input1,
                                                       const half4* __restrict__ input2, const half4* __restrict__ bias,
                                                       int32_t m, int32_t n, const float* __restrict__ weight_amax,
                                                       const float* __restrict__ input1_amax_ptr,
                                                       const int32_t scale_is_vector) {
  int32_t col_start = threadIdx.x << 2;
  const float4* weight_scale_ptr = (const float4*)weight_amax;
  const float weight_scale_single = __ldg(weight_amax);
  const float4 weight_scale = scale_is_vector == 1 ? __ldg(weight_scale_ptr + threadIdx.x * scale_is_vector)
                                                   : make_float4(weight_scale_single, weight_scale_single,
                                                                 weight_scale_single, weight_scale_single);
  const float input1_deQ = __ldg(input1_amax_ptr) / 127.0f;

  float local_out[4];
  int32_t outIdx = ((col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start & 31)) >> 2;
  int4* input1TmpPtr = (int4*)input1;
  int4 input1Tmp = input1TmpPtr[outIdx];
  half4 input2Tmp = input2[outIdx];
  half4 biasTmp = bias[threadIdx.x];

  local_out[0] = static_cast<float>(input2Tmp.x) +
                 static_cast<float>(input1Tmp.x) * input1_deQ * weight_scale.x / 127.0f + static_cast<float>(biasTmp.x);
  local_out[1] = static_cast<float>(input2Tmp.y) +
                 static_cast<float>(input1Tmp.y) * input1_deQ * weight_scale.y / 127.0f + static_cast<float>(biasTmp.y);
  local_out[2] = static_cast<float>(input2Tmp.z) +
                 static_cast<float>(input1Tmp.z) * input1_deQ * weight_scale.z / 127.0f + static_cast<float>(biasTmp.z);
  local_out[3] = static_cast<float>(input2Tmp.w) +
                 static_cast<float>(input1Tmp.w) * input1_deQ * weight_scale.w / 127.0f + static_cast<float>(biasTmp.w);

  half4 outTmp;
  outTmp.x = static_cast<half>(local_out[0]);
  outTmp.y = static_cast<half>(local_out[1]);
  outTmp.z = static_cast<half>(local_out[2]);
  outTmp.w = static_cast<half>(local_out[3]);

  output[outIdx] = outTmp;
}

template <typename T>
void InvokeAddBiasResidualCol32(T* output, const int32_t* input1, const T* input2, const T* bias, int32_t m, int32_t n,
                                cudaStream_t stream, const float* weight_amax, const float* input1_amax_ptr,
                                const int32_t scale_is_vector) {
  dim3 grid(m);
  dim3 block(n >> 2);
  if (block.x > ADD_BIAS_RES_BLOCK_SIZE) {
    throw std::runtime_error("block dim x is bigger than 1024");
  }

  if (sizeof(T) == 2) {
    AddBiasInputCOL32Int32IDataTypeOKernel<<<grid, block, 0, stream>>>((half4*)output, input1, (const half4*)input2,
                                                                       (const half4*)bias, m, n, weight_amax,
                                                                       input1_amax_ptr, scale_is_vector);
  } else {
    AddBiasInputCOL32Int32IDataTypeOKernel<T>
        <<<grid, block, 0, stream>>>(output, input1, input2, bias, m, n, weight_amax, input1_amax_ptr, scale_is_vector);
  }
}

template void InvokeAddBiasResidualCol32(float* output, const int32_t* input1, const float* input2, const float* bias,
                                         int32_t m, int32_t n, cudaStream_t stream, const float* weight_amax,
                                         const float* input1_amax_ptr, const int32_t scale_is_vector);

template void InvokeAddBiasResidualCol32(half* output, const int32_t* input1, const half* input2, const half* bias,
                                         int32_t m, int32_t n, cudaStream_t stream, const float* weight_amax,
                                         const float* input1_amax_ptr, const int32_t scale_is_vector);

}  // namespace nvidia
}  // namespace llm_kernels
