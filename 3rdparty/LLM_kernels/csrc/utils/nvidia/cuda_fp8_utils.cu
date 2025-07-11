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

#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace utils {
#ifdef ENABLE_FP8

template <typename T_OUT, typename T_IN>
__global__ void QuantizeMatrix(T_OUT* output, float const* scale, T_IN const* input, uint32_t num_channels,
                               uint32_t channel_size) {
  uint32_t k = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t n = blockIdx.y;
  if (n < num_channels && k < channel_size) {
    output[n * channel_size + k] =
        (T_OUT)(min(max((float)(input[n * channel_size + k]) / __ldg(scale + n), -FP8_E4M3_MAX), FP8_E4M3_MAX));
  }
}

template <typename T_OUT, typename T_IN>
void InvokeQuantizeMatrix(T_OUT* output, float const* scale, T_IN const* input, uint32_t num_channels,
                          uint32_t channel_size, cudaStream_t stream) {
  dim3 grid((channel_size + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM,
            num_channels);
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  QuantizeMatrix<T_OUT, T_IN><<<grid, block, 0, stream>>>(output, scale, input, num_channels, channel_size);
}

#  define INVOKE_QUANTIZE_MATRIX(type_out, type_in)                                                                    \
    template void InvokeQuantizeMatrix<type_out, type_in>(type_out * output, float const* scale, type_in const* input, \
                                                          uint32_t num_channels, uint32_t channel_size,                \
                                                          cudaStream_t stream);

INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float);
INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, half);
INVOKE_QUANTIZE_MATRIX(half, __nv_fp8_e4m3);
INVOKE_QUANTIZE_MATRIX(float, __nv_fp8_e4m3);
INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, __nv_bfloat16);
INVOKE_QUANTIZE_MATRIX(__nv_bfloat16, __nv_fp8_e4m3);

#  undef INVOKE_QUANTIZE_MATRIX

template <typename T_OUT, typename T_IN, typename T_FAKE>
__global__ void InvokeFakeQuantizeKernel(T_OUT* dst, const T_IN* src, const int32_t size) {
  for (int32_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
    T_FAKE tmp = (T_FAKE)((float)src[tid]);
    dst[tid] = (T_OUT)((float)tmp);
  }
}

template <typename T_OUT, typename T_IN, typename T_FAKE>
void InvokeFakeQuantize(T_OUT* dst, const T_IN* src, const int32_t size, cudaStream_t stream) {
  InvokeFakeQuantizeKernel<T_OUT, T_IN, T_FAKE>
      <<<DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM, DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM, 0, stream>>>(dst, src, size);
}

template void InvokeFakeQuantize<float, float, __nv_fp8_e4m3>(float* dst, const float* src, const int32_t size,
                                                              cudaStream_t stream);
template void InvokeFakeQuantize<half, half, __nv_fp8_e4m3>(half* dst, const half* src, const int32_t size,
                                                            cudaStream_t stream);
template void InvokeFakeQuantize<__nv_bfloat16, __nv_bfloat16, __nv_fp8_e4m3>(__nv_bfloat16* dst,
                                                                              const __nv_bfloat16* src,
                                                                              const int32_t size, cudaStream_t stream);

template <typename T_IN>
__global__ void ComputeFP8QuantizeScaleKernel(float* output, const T_IN* input, const int32_t num_channels,
                                              const int32_t channel_size) {
  const int num_warps = DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM / DEFAULT_CUDA_WARP_SIZE;
  __shared__ float shmem[num_warps];
  int k = threadIdx.x + blockDim.x * blockIdx.x;
  int n = blockIdx.y;
  // min of fabs is 0.f
  float scale = 0.f;
  if (k < channel_size && n < num_channels) {
    float val = fabs((float)(input[n * channel_size + k]));
    scale = fabs(val);
  }
  // warp_reduce
  for (int offset = DEFAULT_CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    scale = max(scale, __shfl_down_sync(0xFFFFFFFF, scale, offset));
  }
  // block_reduce
  if (threadIdx.x % DEFAULT_CUDA_WARP_SIZE == 0) {
    shmem[threadIdx.x / DEFAULT_CUDA_WARP_SIZE] = scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_warps; ++i) {
      scale = max(scale, shmem[i]);
    }
  }
  // grid reduce
  if (threadIdx.x == 0) {
    scale = max(scale / FP8_E4M3_MAX, FP8_E4M3_MIN_SCALE);
    atomicMax(reinterpret_cast<unsigned int*>(output + n), __float_as_uint(scale));
  }
}

template <typename T_IN>
void InvokeComputeFP8QuantizeScale(float* output, const T_IN* input, const int32_t num_channels,
                                   const int32_t channel_size, cudaStream_t stream) {
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  dim3 grid((channel_size + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM,
            num_channels);
  // atomicMax compare origin data in output with new output.
  cudaMemsetAsync(output, 0, sizeof(float) * num_channels, stream);
  ComputeFP8QuantizeScaleKernel<T_IN><<<grid, block, 0, stream>>>(output, input, num_channels, channel_size);
}

template void InvokeComputeFP8QuantizeScale(float* output, const half* input, const int32_t num_channels,
                                            const int32_t channel_size, cudaStream_t stream);

template void InvokeComputeFP8QuantizeScale(float* output, const __nv_bfloat16* input, const int32_t num_channels,
                                            const int32_t channel_size, cudaStream_t stream);

template void InvokeComputeFP8QuantizeScale(float* output, const float* input, const int32_t num_channels,
                                            const int32_t channel_size, cudaStream_t stream);

__global__ void RescaleFp8E4m3Kernel(void* input, void* output, size_t n, const float* input_scale,
                                     const float* output_scale) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    float scale = *input_scale / *output_scale;
    *((__nv_fp8_e4m3*)output + idx) = (__nv_fp8_e4m3)((float)*((__nv_fp8_e4m3*)input + idx) * scale);
  }
}

void InvokeRescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                          cudaStream_t& stream) {
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  dim3 grid((n + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  RescaleFp8E4m3Kernel<<<grid, block, 0, stream>>>(input, output, n, input_scale, output_scale);
}
#endif  // ENABLE_FP8

}  // namespace utils
}  // namespace llm_kernels
