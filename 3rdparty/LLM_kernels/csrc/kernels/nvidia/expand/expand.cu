/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "expand.h"

#include <cuda_runtime.h>
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void expandMatrixKernel(const T* __restrict__ input, T* __restrict__ output, const int m, const int n,
                                   const int expand_num) {
  const int row = blockIdx.x;
  if (row >= m) return;

  extern __shared__ unsigned char sharedMem[];
  T* sharedRow = reinterpret_cast<T*>(sharedMem);
  // put dim n's data into shared memory
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    sharedRow[j] = input[row * n + j];
  }

  __syncthreads();

  for (int e = threadIdx.y; e < expand_num; e += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      output[(row * expand_num + e) * n + j] = sharedRow[j];
    }
  }
}

template <typename T>
struct VecTraits {};

// 4 float = float4
template <>
struct VecTraits<float> {
  using vec4_type = float4;
  static constexpr int vec4_size = 4;
};

// 4 half = uint2
template <>
struct VecTraits<half> {
  using vec4_type = uint2;
  static constexpr int vec4_size = 4;

  __device__ static inline vec4_type load4(const half* ptr) { return *reinterpret_cast<const uint2*>(ptr); }

  __device__ static inline void store4(half* ptr, const vec4_type& val) { *reinterpret_cast<uint2*>(ptr) = val; }
};

// 4 __nv_bfloat16 = uint2
template <>
struct VecTraits<__nv_bfloat16> {
  using vec4_type = uint2;
  static constexpr int vec4_size = 4;

  __device__ static inline vec4_type load4(const __nv_bfloat16* ptr) { return *reinterpret_cast<const uint2*>(ptr); }

  __device__ static inline void store4(__nv_bfloat16* ptr, const vec4_type& val) {
    *reinterpret_cast<uint2*>(ptr) = val;
  }
};

template <typename T>
__global__ void vectorizedExpandMatrixKernel(const T* __restrict__ input, T* __restrict__ output, const int m,
                                             const int n, const int expand_num) {
  using vec4_t = typename VecTraits<T>::vec4_type;
  constexpr int vec_size = VecTraits<T>::vec4_size;

  const int row = blockIdx.x;
  if (row >= m) return;

  const int n_vec = n / vec_size;

  extern __shared__ unsigned char sharedMem[];
  vec4_t* sharedRow = reinterpret_cast<vec4_t*>(sharedMem);
  // put dim n's data into shared memory
  for (int j = threadIdx.x; j < n_vec; j += blockDim.x) {
    if constexpr (std::is_same<T, float>::value) {
      sharedRow[j] = reinterpret_cast<const vec4_t*>(input)[row * n_vec + j];
    } else {
      sharedRow[j] = VecTraits<T>::load4(&input[(row * n + j * vec_size)]);
    }
  }

  __syncthreads();

  for (int e = threadIdx.y; e < expand_num; e += blockDim.y) {
    for (int j = threadIdx.x; j < n_vec; j += blockDim.x) {
      if constexpr (std::is_same<T, float>::value) {
        reinterpret_cast<vec4_t*>(output)[(row * expand_num + e) * n_vec + j] = sharedRow[j];
      } else {
        VecTraits<T>::store4(&output[((row * expand_num + e) * n + j * vec_size)], sharedRow[j]);
      }
    }
  }
}

// only support expand input[m, n] to output[m, expand_num, n]
template <typename T>
void InvokeExpand(const T* input, T* output, const int32_t m, const int32_t expand_num, const int32_t n,
                  const size_t stride, cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(stride == 0, "Expand stride func is temporarily not supported.");

  dim3 blockSize(DEFAULT_CUDA_WARP_SIZE, DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM / DEFAULT_CUDA_WARP_SIZE);
  int gridSize = m;
  // shared memory size
  size_t sharedMemSize = n * sizeof(T);

  if ((n & 3) == 0 && (n >> 2)) {
    size_t vec_shared_size = (n / 4) * sizeof(typename VecTraits<T>::vec4_type);
    vectorizedExpandMatrixKernel<T><<<gridSize, blockSize, vec_shared_size, stream>>>(input, output, m, n, expand_num);
  } else {
    expandMatrixKernel<T><<<gridSize, blockSize, sharedMemSize, stream>>>(input, output, m, n, expand_num);
  }
}

template void InvokeExpand(const float* input, float* output, const int32_t m, const int32_t expand_num,
                           const int32_t n, const size_t stride, cudaStream_t stream);
template void InvokeExpand(const half* input, half* output, const int32_t m, const int32_t expand_num, const int32_t n,
                           const size_t stride, cudaStream_t stream);
template void InvokeExpand(const __nv_bfloat16* input, __nv_bfloat16* output, const int32_t m, const int32_t expand_num,
                           const int32_t n, const size_t stride, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
