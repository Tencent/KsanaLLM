/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <assert.h>
#include <array>
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
#  include <cooperative_groups/reduce.h>
#else
#  include <cooperative_groups.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <type_traits>
#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"

namespace cg = cooperative_groups;

namespace llm_kernels {
namespace nvidia {

template <int32_t VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
  using type = uint16_t;
};
template <>
struct BytesToType<4> {
  using type = uint32_t;
};
template <>
struct BytesToType<8> {
  using type = uint64_t;
};
template <>
struct BytesToType<16> {
  using type = float4;
};

template <int32_t Bytes>
__device__ inline void copy(const void* local, void* data) {
  using T = typename BytesToType<Bytes>::type;

  const T* in = static_cast<const T*>(local);
  T* out = static_cast<T*>(data);
  *out = *in;
}

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int32_t mask = 16; mask > 0; mask >>= 1)
    //__shfl_sync bf16 return float when sm < 80
    val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T BlockReduceSum(T val) {
  static __shared__ T shared[32];
  int32_t lane = threadIdx.x & 0x1f;
  int32_t wid = threadIdx.x >> 5;

  val = WarpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = WarpReduceSum<T>(val);

  return val;
}

template <typename T>
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
  for (int32_t mask = 16; mask > 0; mask >>= 1) val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

// Calculate the maximum of all elements in a block
template <typename T>
__inline__ __device__ T BlockReduceMax(T val) {
  static __shared__ T shared[32];
  int32_t lane = threadIdx.x & 0x1f;  // in-warp idx
  int32_t wid = threadIdx.x >> 5;     // warp idx

  val = WarpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = WarpReduceMax(val);

  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T BlockAllReduceMax(T val) {
  static __shared__ T shared[32];
  int32_t lane = threadIdx.x & 0x1f;  // in-warp idx
  int32_t wid = threadIdx.x >> 5;     // warp idx

  val = WarpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = WarpReduceMax(val);

  return val;
}

template <typename T, int32_t NUM>
__inline__ __device__ T WarpReduceSumV2(T* val) {
#pragma unroll
  for (int32_t i = 0; i < NUM; i++) {
#pragma unroll
    for (int32_t mask = 16; mask > 0; mask >>= 1) val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int32_t NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int32_t lane = threadIdx.x & 0x1f;
  int32_t wid = threadIdx.x >> 5;

  WarpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int32_t i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  WarpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <typename T, int32_t NUM>
__inline__ __device__ T WarpReduceMaxV2(T* val) {
#pragma unroll
  for (int32_t i = 0; i < NUM; i++) {
#pragma unroll
    for (int32_t mask = 16; mask > 0; mask >>= 1) val[i] = max(val[i], __shfl_xor_sync(FINAL_MASK, val[i], mask, 32));
  }
  return (T)(0.0f);
}

template <typename T, int32_t NUM>
__inline__ __device__ T blockReduceMaxV2(T* val) {
  static __shared__ T shared[32][NUM];
  int32_t lane = threadIdx.x & 0x1f;  // in-warp idx
  int32_t wid = threadIdx.x >> 5;     // warp idx

  WarpReduceMaxV2<T, NUM>(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
  {
#pragma unroll
    for (int32_t i = 0; i < NUM; i++) {
      shared[wid][i] = val[i];
    }
  }

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int32_t i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[lane][i] : (T)-1e20f;
  }
  WarpReduceMaxV2<T, NUM>(val);

  return (T)0.0f;
}

template <int32_t NUM>
__inline__ __device__ void CooperateGroupBlockReduceSumElements(float* element_list,
                                                                float* cgBlockReduceSumElements_shm) {
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

  const int32_t tid = cta.thread_rank();
  const int32_t blockz = blockDim.x;
  for (int32_t i = 0; i < NUM; i++) {
#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))
    cgBlockReduceSumElements_shm[i * blockz + tid] = cg::reduce(tile, element_list[i], cg::plus<float>());
#else
    // TODO Add implementation here
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("[ERROR] Not support CooperateGroupBlockReduceSumElements when CUDA < 11 \n");
      assert(false);
    }
#endif
  }
  cg::sync(cta);
  if (tid == 0) {
#pragma unroll
    for (int32_t i = 0; i < NUM; i++) {
      float beta = 0.0f;
      for (int32_t j = 0; j < blockz; j += 32) {
        beta += cgBlockReduceSumElements_shm[i * blockz + j];
      }
      element_list[i] = beta;
    }
  }
}

template <typename T, int32_t MAX_K>
struct TopK {
  int32_t p[MAX_K];
  T u[MAX_K];

  __device__ __forceinline__ void insert(T elem, int32_t elem_id) {
    if (elem > u[MAX_K - 1] || (p[MAX_K - 1] == -1) || ((elem == u[MAX_K - 1]) && (elem_id < p[MAX_K - 1]))) {
      u[MAX_K - 1] = elem;
      p[MAX_K - 1] = elem_id;
    }

    for (int32_t k = MAX_K - 2; k >= 0; --k) {
      if ((u[k + 1] > u[k]) || (p[k] == -1) || ((u[k + 1] == u[k]) && (p[k + 1] < p[k]))) {
        T u2 = u[k];
        int32_t p2 = p[k];
        u[k] = u[k + 1];
        p[k] = p[k + 1];
        u[k + 1] = u2;
        p[k + 1] = p2;
      }
    }
  }

  __device__ __forceinline__ void init() {
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    for (int32_t i = 0; i < MAX_K; i++) {
      p[i] = -1;
      u[i] = -MAX_T_VAL;
    }
  }
};

template <typename T, int32_t MAX_K>
__device__ __forceinline__ TopK<T, MAX_K> reduce_topk_op(const TopK<T, MAX_K>& a, const TopK<T, MAX_K>& b) {
  TopK<T, MAX_K> res = a;
  for (int32_t i = 0; i < MAX_K; ++i) res.insert(b.u[i], b.p[i]);
  return res;
}

template <typename T>
struct TopK_2 {
  int32_t p = -1;
  T u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);

  __device__ __forceinline__ void insert(T elem, int32_t elem_id) {
    if (elem > u) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
    p = -1;
  }
};

template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a, const TopK_2<T>& b) {
  return a.u > b.u ? a : b;
}

template <typename T>
__device__ __forceinline__ T ClampInfForHalf(const float input) {
  return input;
}

template <>
__device__ __forceinline__ half ClampInfForHalf(const float input) {
  // clamp inf values to enable fp16 training
  return input > 0.0f ? (half)min(input, HALF_FLT_MAX - 1000) : (half)max(input, -HALF_FLT_MAX + 1000);
}

template <>
__device__ __forceinline__ __nv_bfloat16 ClampInfForHalf(const float input) {
  return __float2bfloat16(input);
}

}  // namespace nvidia
}  // namespace llm_kernels
