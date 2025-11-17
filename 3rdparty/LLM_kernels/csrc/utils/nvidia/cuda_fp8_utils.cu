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

#include "csrc/kernels/nvidia/common/reduce_kernel_utils.cuh"

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"

namespace llm_kernels {
namespace utils {
#ifdef ENABLE_FP8

// Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
template <typename T_OUT, typename T_IN>
__global__ void QuantizeMatrix(T_OUT* output, float const* scale, T_IN const* input, size_t num_channels,
                               size_t channel_size) {
  size_t element_index = threadIdx.x + blockIdx.x * blockDim.x;
  size_t channel_index = blockIdx.y;
  if (channel_index < num_channels && element_index < channel_size) {
    output[channel_index * channel_size + element_index] =
        (T_OUT)(min(max((float)(input[channel_index * channel_size + element_index]) / __ldg(scale + channel_index), -FP8_E4M3_MAX), FP8_E4M3_MAX));
  }
}

/*
 * Adapted from
 * [sglang-project]
 * https://github.com/sgl-project/sglang/blob/v0.4.9/sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu
 * The input address of each token must be aligned to 16-byte.
 * Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
 */
template <typename T_OUT, typename T_IN>
__global__ void QuantizeMatrix_v2(T_OUT* output, float const* scale, T_IN const* input, size_t num_channels,
                               size_t channel_size) {
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t channel_index = blockIdx.y;
  const size_t grid_size = blockDim.x * gridDim.x;
  const float scale_val = __ldg(scale + channel_index);

  T_IN const* input_ptr = input + channel_index * channel_size;
  T_OUT* output_ptr = output + channel_index * channel_size;

  constexpr size_t elements_per_vector = VEC_MEM_ACC_SIZE;
  using vec_t = llm_kernels::nvidia::vec_t<T_IN, elements_per_vector>;
  const size_t num_vec_elems = channel_size / elements_per_vector;

  // 16-byte write loop configuration, auto-optimized for different output types.
  constexpr size_t store_loop_num = elements_per_vector * sizeof(T_OUT) / VEC_MEM_ACC_SIZE;

  for (size_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input_ptr + i * elements_per_vector);

    alignas(VEC_MEM_ACC_SIZE) T_OUT output_buffer[elements_per_vector];
#pragma unroll
    for (size_t j = 0; j < elements_per_vector; ++j) {
      float val = fminf(FP8_E4M3_MAX, fmaxf(-FP8_E4M3_MAX, static_cast<float>(input_vec[j]) / scale_val));
      output_buffer[j] = static_cast<T_OUT>(val);
    }

#pragma unroll
    for (size_t loop_idx = 0; loop_idx < store_loop_num; ++loop_idx) {
      // Use uint4 for 16-byte batched writes to maximize memory bandwidth.
      *(uint4*)((char*)(output_ptr + i * elements_per_vector) + loop_idx * VEC_MEM_ACC_SIZE) = *(uint4*)((char*)output_buffer + loop_idx * VEC_MEM_ACC_SIZE);
    }
  }

  // Handle remaining elements: scalar processing for tail data that cannot be vectorized.
  const size_t remaining_start = num_vec_elems * elements_per_vector;
  for (size_t idx = remaining_start + gid; idx < channel_size; idx += grid_size) {
    float val = fminf(FP8_E4M3_MAX, fmaxf(-FP8_E4M3_MAX, static_cast<float>(input_ptr[idx]) / scale_val));
    output_ptr[idx] = static_cast<T_OUT>(val);
  }
}

// Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
template <typename T_OUT, typename T_IN>
void InvokeQuantizeMatrix(T_OUT* output, float const* scale, T_IN const* input, size_t num_channels,
                          size_t channel_size, cudaStream_t stream) {
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);

  constexpr size_t elements_per_vector = VEC_MEM_ACC_SIZE;
  const size_t num_blocks_per_token = min(((channel_size + (elements_per_vector - 1)) / elements_per_vector + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM, (size_t)1024);

  static int32_t sm_count = GetSMCount();
  // NOTE(ryanyhuang): Minimum number of active blocks required, determined through testing.
  // This value is set to one-fourth of the total SM count.
  static size_t min_required_active_blocks = sm_count >> 2;

  // Check if we can use the vectorized version or need to switch to the original version.
  if (num_blocks_per_token * num_channels >= min_required_active_blocks) {
    // Use vectorized kernel for better performance.
    dim3 grid(num_blocks_per_token, num_channels);
    QuantizeMatrix_v2<T_OUT, T_IN><<<grid, block, 0, stream>>>(output, scale, input, num_channels, channel_size);
  } else {
    dim3 grid((channel_size + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM,
            num_channels);
    QuantizeMatrix<T_OUT, T_IN><<<grid, block, 0, stream>>>(output, scale, input, num_channels, channel_size);
  }
}

#  define INVOKE_QUANTIZE_MATRIX(type_out, type_in)                                                                    \
    template void InvokeQuantizeMatrix<type_out, type_in>(type_out * output, float const* scale, type_in const* input, \
                                                          size_t num_channels, size_t channel_size, cudaStream_t stream);

INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, float);
INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, half);
INVOKE_QUANTIZE_MATRIX(__nv_fp8_e4m3, __nv_bfloat16);

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

// Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
template <typename T_IN>
__global__ void ComputeFP8QuantizeScaleKernel(float* output, const T_IN* input, const int32_t num_channels,
                                              const int32_t channel_size) {
  const int num_warps = DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM / DEFAULT_CUDA_WARP_SIZE;
  __shared__ float shmem[num_warps];
  int element_index = threadIdx.x + blockDim.x * blockIdx.x;
  int channel_index = blockIdx.y;
  // min of fabs is 0.f
  float scale = 0.f;
  if (element_index < channel_size && channel_index < num_channels) {
    float val = fabs((float)(input[channel_index * channel_size + element_index]));
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
    atomicMax(reinterpret_cast<unsigned int*>(output + channel_index), __float_as_uint(scale));
  }
}

/*
 * Adapted from
 * [sglang-project]
 * https://github.com/sgl-project/sglang/blob/v0.4.9/sgl-kernel/csrc/gemm/per_tensor_quant_fp8.cu
 * The input address of each token must be aligned to 16-byte.
 * Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
 */
template <typename T_IN>
__global__ void ComputeFP8QuantizeScaleKernel_v2(float* output, const T_IN* input, const int32_t num_channels,
                                              const int32_t channel_size) {
  float max_value = 0.0f;
  const size_t tid = threadIdx.x;
  const size_t channel_index = blockIdx.y;
  const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_size = blockDim.x * gridDim.x;

  const T_IN* input_ptr = input + channel_index * channel_size;
  float* output_ptr = output + channel_index;

  // Vectorization configuration: auto-adjust vector size based on input type.
  constexpr size_t elements_per_vector = VEC_MEM_ACC_SIZE / sizeof(T_IN);
  using vec_t = llm_kernels::nvidia::vec_t<T_IN, elements_per_vector>;

  const size_t num_vec_elems = channel_size / elements_per_vector;

  for (size_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input_ptr + i * elements_per_vector);

#pragma unroll
    for (size_t j = 0; j < elements_per_vector; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  // Handle remaining elements: scalar processing for tail data that cannot be vectorized.
  const size_t remaining_start = num_vec_elems * elements_per_vector;
  for (size_t idx = remaining_start + gid; idx < channel_size; idx += grid_size) {
    float val = static_cast<float>(input_ptr[idx]);
    max_value = fmaxf(max_value, fabsf(val));
  }

  max_value = llm_kernels::nvidia::BlockReduceMax<float>(max_value);

  // Write final quantization scale, ensuring it's within FP8 valid range.
  if (tid == 0) {
    max_value = max(max_value / FP8_E4M3_MAX, FP8_E4M3_MIN_SCALE);
    atomicMax(reinterpret_cast<unsigned int*>(output_ptr), __float_as_uint(max_value));
  }
}

// Only support FP8 E4M3 format (8-bit floating point: 4 exponent bits, 3 mantissa bits).
template <typename T_IN>
void InvokeComputeFP8QuantizeScale(float* output, const T_IN* input, const int32_t num_channels,
                                   const int32_t channel_size, cudaStream_t stream) {
  dim3 block(DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);

  // Init data: atomicMax will compare origin data in output with new output.
  cudaMemsetAsync(output, 0, sizeof(float) * num_channels, stream);

  constexpr size_t elements_per_vector = VEC_MEM_ACC_SIZE / sizeof(T_IN);
  const size_t num_blocks_per_token = min(((channel_size + (elements_per_vector - 1)) / elements_per_vector + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM, (size_t)1024);

  static int32_t sm_count = GetSMCount();
  // NOTE(ryanyhuang): Minimum number of active blocks required, determined through testing.
  // This value is set to one-fourth of the total SM count.
  static size_t min_required_active_blocks = sm_count >> 2;

  // Check if we can use the vectorized version or need to switch to the original version.
  if (num_blocks_per_token * num_channels >= min_required_active_blocks) {
    // Use vectorized kernel for better performance。
    dim3 grid(num_blocks_per_token, num_channels);
    ComputeFP8QuantizeScaleKernel_v2<T_IN><<<grid, block, 0, stream>>>(output, input, num_channels, channel_size);
  } else {
    dim3 grid((channel_size + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM,
            num_channels);
    ComputeFP8QuantizeScaleKernel<T_IN><<<grid, block, 0, stream>>>(output, input, num_channels, channel_size);
  }
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
