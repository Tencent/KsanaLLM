/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/concat/concat.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void ConcatVectorizedKernel(const T* __restrict__ input_a, const T* __restrict__ input_b,
                                       const size_t concat_size_a, const size_t concat_size_b,
                                       const size_t outer_dim_size, const size_t inner_dim_size,
                                       const size_t kPackElems, T* __restrict__ output) {
  const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * kPackElems;
  const size_t concat_dim_size_sum = concat_size_a + concat_size_b;
  if (idx >= outer_dim_size * concat_dim_size_sum * inner_dim_size) {
    return;
  }

  const size_t concat_inner_size = concat_dim_size_sum * inner_dim_size;
  const size_t outer_idx = idx / concat_inner_size;
  const size_t concat_idx = (idx % concat_inner_size) / inner_dim_size;
  const size_t inner_idx = idx % inner_dim_size;

  if (concat_idx < concat_size_a) {
    const size_t offset = outer_idx * concat_size_a * inner_dim_size + concat_idx * inner_dim_size + inner_idx;
    output[idx / kPackElems] = input_a[offset / kPackElems];
  } else {
    const size_t offset =
        outer_idx * concat_size_b * inner_dim_size + (concat_idx - concat_size_a) * inner_dim_size + inner_idx;
    output[idx / kPackElems] = input_b[offset / kPackElems];
  }
}

template <typename T>
void Concat(const T* __restrict__ input_a, const T* __restrict__ input_b, size_t concat_size_a, size_t concat_size_b,
            size_t outer_dim_size, size_t inner_dim_size, T* __restrict__ output, cudaStream_t& stream) {
  const size_t total_elements = outer_dim_size * (concat_size_a + concat_size_b) * inner_dim_size;
  const dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);

  const size_t kSizeT = sizeof(T);
  const size_t size_a = concat_size_a * inner_dim_size * kSizeT;
  const size_t size_b = concat_size_b * inner_dim_size * kSizeT;
  if (size_a % 16 == 0 && size_b % 16 == 0) {
    int elements_num = 16 / kSizeT;
    const dim3 grid(
        (total_elements / elements_num + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM, 1);
    using VecType = typename utils::PackType<float, 4>::type;

    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b), concat_size_a,
        concat_size_b, outer_dim_size, inner_dim_size, elements_num, reinterpret_cast<VecType*>(output));
  } else if (size_a % 8 == 0 && size_b % 8 == 0) {
    int elements_num = 8 / kSizeT;
    const dim3 grid(
        (total_elements / elements_num + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM, 1);
    using VecType = typename utils::PackType<float, 2>::type;

    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b), concat_size_a,
        concat_size_b, outer_dim_size, inner_dim_size, elements_num, reinterpret_cast<VecType*>(output));
  } else if (size_a % 4 == 0 && size_b % 4 == 0) {
    int elements_num = 4 / kSizeT;
    const dim3 grid(
        (total_elements / elements_num + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM, 1);
    using VecType = typename utils::PackType<float, 1>::type;

    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b), concat_size_a,
        concat_size_b, outer_dim_size, inner_dim_size, elements_num, reinterpret_cast<VecType*>(output));
  } else if (size_a % 2 == 0 && size_b % 2 == 0) {
    int elements_num = 2 / kSizeT;
    const dim3 grid(
        (total_elements / elements_num + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM, 1);
    using VecType = typename utils::PackType<half, 1>::type;

    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b), concat_size_a,
        concat_size_b, outer_dim_size, inner_dim_size, elements_num, reinterpret_cast<VecType*>(output));
  } else {
    const dim3 grid((total_elements + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM, 1);
    using VecType = typename utils::PackType<T, 1>::type;

    ConcatVectorizedKernel<VecType><<<grid, block, 0, stream>>>(
        reinterpret_cast<const VecType*>(input_a), reinterpret_cast<const VecType*>(input_b), concat_size_a,
        concat_size_b, outer_dim_size, inner_dim_size, 1, reinterpret_cast<VecType*>(output));
  }
}

#define INSTANTIATE_CONCAT(T)                                                                                      \
  template void Concat(const T* __restrict__ input_a, const T* __restrict__ input_b, size_t concat_size_a,         \
                       size_t concat_size_b, size_t outer_dim_size, size_t inner_dim_size, T* __restrict__ output, \
                       cudaStream_t& stream);

INSTANTIATE_CONCAT(float);
INSTANTIATE_CONCAT(half);
#ifdef ENABLE_BF16
INSTANTIATE_CONCAT(__nv_bfloat16);
#endif

#undef INSTANTIATE_CONCAT

}  // namespace nvidia
}  // namespace llm_kernels
