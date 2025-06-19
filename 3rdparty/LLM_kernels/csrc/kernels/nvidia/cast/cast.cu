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

#include "cast.h"

#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

static constexpr size_t MAX_BLOCKS_NUM = 1024;
static constexpr size_t DEFAULT_THREADS_NUM = 256;
static constexpr size_t DEFAULT_WARP_THREADS_NUM = 32;

__global__ void ConvertInt64ToIntKernel(const int64_t* input, int* output, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    output[idx] = static_cast<int>(input[idx]);
  }
}

void Int64ToInt(const int64_t* input, size_t input_length, int* output, cudaStream_t& stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertInt64ToIntKernel<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFloatToHalfKernel(const float* input, half* output, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float val = input[idx];
    half halfVal = __float2half(val);
    output[idx] = halfVal;
  }
}

void FloatToHalf(const float* input, size_t input_length, half* output, cudaStream_t stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertFloatToHalfKernel<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFloatToBFloat16Kernel(const float* input, __nv_bfloat16* output, size_t size) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float val = input[idx];
    __nv_bfloat16 bfloat16Val = __float2bfloat16(val);
    output[idx] = bfloat16Val;
  }
}

void FloatToBFloat16(const float* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertFloatToBFloat16Kernel<<<grid, block, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertHalfToFloatKernel(const half* input, float* output, size_t input_length,
                                         const size_t input_stride = 1, const size_t output_stride = 1) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < input_length) {
    size_t batch = idx / input_stride;
    size_t input_idx = idx % input_stride;
    if (input_idx >= output_stride) {
      return;
    }
    half val = input[idx];
    float floatVal = __half2float(val);
    output[batch * output_stride + input_idx] = floatVal;
  }
}

void HalfToFloat(const half* input, size_t input_length, float* output, cudaStream_t stream, const size_t input_stride,
                 const size_t output_stride) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertHalfToFloatKernel<<<grid, block, 0, stream>>>(input, output, input_length, input_stride, output_stride);
}

__global__ void ConvertBFloat16ToFloatKernel(const __nv_bfloat16* input, float* output, size_t input_length,
                                             const size_t input_stride = 1, const size_t output_stride = 1) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < input_length) {
    size_t batch = idx / input_stride;
    size_t input_idx = idx % input_stride;
    if (input_idx >= output_stride) {
      return;
    }
    __nv_bfloat16 val = input[idx];
    float floatVal = __bfloat162float(val);
    output[batch * output_stride + input_idx] = floatVal;
  }
}

void BFloat16ToFloat(const __nv_bfloat16* input, size_t input_length, float* output, cudaStream_t stream,
                     const size_t input_stride, const size_t output_stride) {
  dim3 grid((input_length + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  ConvertBFloat16ToFloatKernel<<<grid, block, 0, stream>>>(input, output, input_length, input_stride, output_stride);
}

__global__ void ConvertHalfToBFloat16Kernel(half* input_output, size_t size) {
  extern __shared__ float shared_mem[];
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int grid_size = blockDim.x * gridDim.x;
  const int stride = grid_size;
  for (size_t i = blockIdx.x * block_size + tid; i < size; i += stride) {
    if (i < size) {
      half input_val = input_output[i];
      float float_val = __half2float(input_val);
      shared_mem[tid] = float_val;
      __syncthreads();
      __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(input_output);
      output_ptr[i] = __float2bfloat16(shared_mem[tid]);
      __syncthreads();
    }
  }
}

void HalfToBFloat16(void* data_ptr, size_t input_length, cudaStream_t stream) {
  int block_size = (input_length < DEFAULT_THREADS_NUM) ? DEFAULT_WARP_THREADS_NUM : DEFAULT_THREADS_NUM;
  int grid_size = min((input_length + block_size - 1) / block_size, MAX_BLOCKS_NUM);
  grid_size = max(1, grid_size);
  dim3 grid(grid_size);
  dim3 block(block_size);
  size_t shared_mem_size = block_size * sizeof(float);
  ConvertHalfToBFloat16Kernel<<<grid, block, shared_mem_size, stream>>>(reinterpret_cast<half*>(data_ptr),
                                                                        input_length);
}

__global__ void ConvertBFloat16ToHalfKernel(__nv_bfloat16* input_output, size_t size) {
  extern __shared__ float shared_mem[];
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int grid_size = blockDim.x * gridDim.x;
  const int stride = grid_size;

  for (size_t i = blockIdx.x * block_size + tid; i < size; i += stride) {
    if (i < size) {
      __nv_bfloat16 input_val = input_output[i];
      float float_val = __bfloat162float(input_val);
      shared_mem[tid] = float_val;
      __syncthreads();
      half* output_ptr = reinterpret_cast<half*>(input_output);
      output_ptr[i] = __float2half(shared_mem[tid]);
      __syncthreads();
    }
  }
}

void BFloat16ToHalf(void* data_ptr, size_t input_length, cudaStream_t stream) {
  int block_size = (input_length < DEFAULT_THREADS_NUM) ? DEFAULT_WARP_THREADS_NUM : DEFAULT_THREADS_NUM;
  int grid_size = min((input_length + block_size - 1) / block_size, MAX_BLOCKS_NUM);
  grid_size = max(1, grid_size);
  dim3 grid(grid_size);
  dim3 block(block_size);
  size_t shared_mem_size = block_size * sizeof(float);
  ConvertBFloat16ToHalfKernel<<<grid, block, shared_mem_size, stream>>>(reinterpret_cast<__nv_bfloat16*>(data_ptr),
                                                                        input_length);
}

#if defined(ENABLE_FP8)
__global__ void ConvertFloatToFp8E4M3Kernel(const float* input, __nv_fp8_e4m3* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __nv_fp8_e4m3(input[idx]);
  }
}

void FloatToFp8E4M3(const float* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertFloatToFp8E4M3Kernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFp8E4M3toFloatKernel(const __nv_fp8_e4m3* input, float* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = float(input[idx]);
  }
}

void Fp8E4M3ToFloat(const __nv_fp8_e4m3* input, size_t input_length, float* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertFp8E4M3toFloatKernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertHalfToFp8E4M3Kernel(const half* input, __nv_fp8_e4m3* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __nv_fp8_e4m3(float(input[idx]));
  }
}

void HalfToFp8E4M3(const half* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertHalfToFp8E4M3Kernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFp8E4M3toHalfKernel(const __nv_fp8_e4m3* input, half* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2half(float(input[idx]));
  }
}

void Fp8E4M3ToHalf(const __nv_fp8_e4m3* input, size_t input_length, half* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertFp8E4M3toHalfKernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertBFloat16ToFp8E4M3Kernel(const __nv_bfloat16* input, __nv_fp8_e4m3* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __nv_fp8_e4m3(float(input[idx]));
  }
}

void BFloat16ToFp8E4M3(const __nv_bfloat16* input, size_t input_length, __nv_fp8_e4m3* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertBFloat16ToFp8E4M3Kernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}

__global__ void ConvertFp8E4M3toBFloat16Kernel(const __nv_fp8_e4m3* input, __nv_bfloat16* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __float2bfloat16(float(input[idx]));
  }
}

void Fp8E4M3ToBFloat16(const __nv_fp8_e4m3* input, size_t input_length, __nv_bfloat16* output, cudaStream_t stream) {
  int num_blocks = (input_length + DEFAULT_THREADS_NUM - 1) / DEFAULT_THREADS_NUM;
  ConvertFp8E4M3toBFloat16Kernel<<<num_blocks, DEFAULT_THREADS_NUM, 0, stream>>>(input, output, input_length);
}
#endif

}  // namespace nvidia
}  // namespace llm_kernels
