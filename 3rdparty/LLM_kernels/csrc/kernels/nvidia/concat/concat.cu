/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/concat/concat.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void ConcatVectorizedKernel(const T* const __restrict__ input_a, const T* const __restrict__ input_b,
                                       const size_t concat_inner_a, const size_t concat_inner_b,
                                       const size_t concat_inner_size, const size_t total_size,
                                       T* const __restrict__ output) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size) {
    return;
  }

  const size_t outer_idx = idx / concat_inner_size;
  const size_t concat_idx = idx % concat_inner_size;
  const bool is_a = concat_idx < concat_inner_a;
  const size_t offset =
      is_a ? outer_idx * concat_inner_a + concat_idx : outer_idx * concat_inner_b + concat_idx - concat_inner_a;

  output[idx] = (is_a ? input_a : input_b)[offset];
}

template <typename T>
void Concat(const T* __restrict__ input_a, const T* __restrict__ input_b, size_t concat_size_a, size_t concat_size_b,
            size_t outer_dim_size, size_t inner_dim_size, T* __restrict__ output, cudaStream_t& stream) {
  constexpr int32_t kThreadNum = DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM;
  const dim3 block(kThreadNum);

  const size_t concat_inner_a = concat_size_a * inner_dim_size;
  const size_t concat_inner_b = concat_size_b * inner_dim_size;
  const size_t concat_inner_size = concat_inner_a + concat_inner_b;
  const size_t total_elements = outer_dim_size * (concat_size_a + concat_size_b) * inner_dim_size;

  constexpr size_t kSizeT = sizeof(T);
  const size_t size_a = concat_inner_a * kSizeT;
  const size_t size_b = concat_inner_b * kSizeT;

  if (size_a % 16 == 0 && size_b % 16 == 0) {
    constexpr int kElementNum = 16 / kSizeT;
    const dim3 grid((total_elements / kElementNum + kThreadNum - 1) / kThreadNum);
    using VecType = typename utils::PackType<float, 4>::type;
    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b),
        concat_inner_a / kElementNum, concat_inner_b / kElementNum, concat_inner_size / kElementNum,
        total_elements / kElementNum, reinterpret_cast<VecType*>(output));
  } else if (size_a % 8 == 0 && size_b % 8 == 0) {
    constexpr int kElementNum = 8 / kSizeT;
    const dim3 grid((total_elements / kElementNum + kThreadNum - 1) / kThreadNum);
    using VecType = typename utils::PackType<float, 2>::type;
    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b),
        concat_inner_a / kElementNum, concat_inner_b / kElementNum, concat_inner_size / kElementNum,
        total_elements / kElementNum, reinterpret_cast<VecType*>(output));
  } else if (size_a % 4 == 0 && size_b % 4 == 0) {
    constexpr int kElementNum = 4 / kSizeT;
    const dim3 grid((total_elements / kElementNum + kThreadNum - 1) / kThreadNum);
    using VecType = typename utils::PackType<float, 1>::type;
    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b),
        concat_inner_a / kElementNum, concat_inner_b / kElementNum, concat_inner_size / kElementNum,
        total_elements / kElementNum, reinterpret_cast<VecType*>(output));
  } else if (size_a % 2 == 0 && size_b % 2 == 0) {
    constexpr int kElementNum = 2 / kSizeT;
    const dim3 grid((total_elements / kElementNum + kThreadNum - 1) / kThreadNum);
    using VecType = typename utils::PackType<half, 1>::type;
    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b),
        concat_inner_a / kElementNum, concat_inner_b / kElementNum, concat_inner_size / kElementNum,
        total_elements / kElementNum, reinterpret_cast<VecType*>(output));
  } else {
    const dim3 grid((total_elements + kThreadNum - 1) / kThreadNum);
    using VecType = typename utils::PackType<T, 1>::type;
    constexpr int kElementNum = 1;
    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b),
        concat_inner_a / kElementNum, concat_inner_b / kElementNum, concat_inner_size / kElementNum,
        total_elements / kElementNum, reinterpret_cast<VecType*>(output));
  }
}

#define INSTANTIATE_CONCAT(T)                                                                                      \
  template void Concat(const T* __restrict__ input_a, const T* __restrict__ input_b, size_t concat_size_a,         \
                       size_t concat_size_b, size_t outer_dim_size, size_t inner_dim_size, T* __restrict__ output, \
                       cudaStream_t& stream);

INSTANTIATE_CONCAT(float);
INSTANTIATE_CONCAT(half);
INSTANTIATE_CONCAT(__nv_bfloat16);

#undef INSTANTIATE_CONCAT

}  // namespace nvidia
}  // namespace llm_kernels
