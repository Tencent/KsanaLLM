/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/adjust_mem/adjust_mem.h"

#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;
namespace llm_kernels {
namespace nvidia {

template <typename T>
__global__ void GatherSubmatrixKernel(const T* __restrict__ input, T* __restrict__ outputs,
                                      const size_t* __restrict__ m_offsets, const size_t* m_num_per_group,
                                      size_t group_size, size_t n_start, size_t n_end, size_t for_range, size_t n) {
  size_t group_id = blockIdx.x;
  size_t row_id_in_group = blockIdx.y;
  if (row_id_in_group >= m_num_per_group[group_id]) {
    return;
  }
  size_t output_row_id = row_id_in_group + m_offsets[group_id];
  size_t col_start_id = threadIdx.x * for_range + n_start;

#pragma unroll
  for (size_t offset = 0; offset < for_range && col_start_id + offset < n_end; ++offset) {
    size_t col_id = col_start_id + offset;
    outputs[output_row_id * (n_end - n_start) + (col_id - n_start)] =
        input[(group_id * group_size + row_id_in_group) * n + col_id];
  }
}

template <typename T>
void InvokeGatherSubmatrix(const T* __restrict__ input, T* __restrict__ output, std::vector<size_t>& m_num_per_group,
                           size_t group_size, size_t group_num, size_t n_start, size_t n_end, size_t m, size_t n,
                           void* workspace, cudaStream_t& stream) {
  assert(group_size * group_num == m);

  using PT = typename PackTypeAlign<T>::type;
  int32_t packed_elems = ElemsNum<PT>::value;
  if ((n_end - n_start) % packed_elems != 0 || n_start % packed_elems != 0 || n_end % packed_elems != 0) {
    packed_elems = 1;
  }

  std::vector<size_t> m_offsets(group_num + 1, 0);
  for (size_t i = 0; i < group_num; ++i) {
    m_offsets[i + 1] = m_offsets[i] + m_num_per_group[i];
  }

  // TODO(rockcao): fuse two memcpy calls into one
  void* device_m_offsets = workspace;
  cudaMemcpyAsync(device_m_offsets, reinterpret_cast<void*>(m_offsets.data()), m_offsets.size() * sizeof(size_t),
                  cudaMemcpyHostToDevice, stream);
  void* device_m_num_per_group = device_m_offsets + m_offsets.size() * sizeof(size_t);
  cudaMemcpyAsync(device_m_num_per_group, reinterpret_cast<void*>(m_num_per_group.data()),
                  m_num_per_group.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);

  const size_t BLOCK_SIZE = 256;
  dim3 grid(group_num, group_size), block(BLOCK_SIZE);
  size_t for_range = ceil(static_cast<float>(n_end - n_start) / BLOCK_SIZE);
  if (packed_elems > 1) {
    GatherSubmatrixKernel<PT><<<grid, block, 0, stream>>>(
        reinterpret_cast<const PT*>(input), reinterpret_cast<PT*>(output), static_cast<const size_t*>(device_m_offsets),
        static_cast<const size_t*>(device_m_num_per_group), group_size, n_start / packed_elems, n_end / packed_elems,
        for_range / packed_elems, n / packed_elems);
  } else {
    GatherSubmatrixKernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), static_cast<const size_t*>(device_m_offsets),
        static_cast<const size_t*>(device_m_num_per_group), group_size, n_start, n_end, for_range, n);
  }
}

#define INSTANTIATE_GATHER_SUBMATRIX(T)                                                                             \
  template void InvokeGatherSubmatrix(                                                                              \
      const T* __restrict__ input, T* __restrict__ output, std::vector<size_t>& m_num_per_group, size_t group_size, \
      size_t group_num, size_t n_start, size_t n_end, size_t m, size_t n, void* workspace, cudaStream_t& stream);

INSTANTIATE_GATHER_SUBMATRIX(float);
INSTANTIATE_GATHER_SUBMATRIX(half);
INSTANTIATE_GATHER_SUBMATRIX(__nv_bfloat16);

#undef INSTANTIATE_GATHER_SUBMATRIX

template <typename T>
__global__ void DpMapCopyKernel(const T* __restrict__ input, T* __restrict__ output,
                                const size_t* __restrict__ group_info, size_t group_num, size_t n, size_t for_range) {
  size_t row_id = blockIdx.x;
  int group_id = -1;

  for (int i = 0; i < group_num; ++i) {
    if (row_id < group_info[4 * i]) {
      group_id = i;
      break;
    }
  }

  size_t row_id_in_group = row_id;
  if (group_id > 0) {
    row_id_in_group = row_id - group_info[4 * (group_id - 1)];
  }
  size_t prefill_row_num_in_group = group_info[4 * group_id + 1];
  size_t output_row_id;

  if (row_id_in_group < prefill_row_num_in_group) {
    output_row_id = group_info[4 * group_id + 2] + row_id_in_group;
  } else {
    output_row_id = group_info[4 * group_id + 3] + row_id_in_group - prefill_row_num_in_group;
  }

  // copy
  size_t col_start_id = threadIdx.x * for_range;
#pragma unroll
  for (size_t offset = 0; offset < for_range && col_start_id + offset < n; ++offset) {
    output[output_row_id * n + col_start_id + offset] = input[row_id * n + col_start_id + offset];
  }
}

template <typename T>
void InvokeDpMapCopy(const T* __restrict__ input, T* __restrict__ output, const std::vector<size_t>& group_info,
                     size_t m, size_t n, void* workspace, cudaStream_t& stream) {
  using PT = typename PackTypeAlign<T>::type;
  int32_t packed_elems = ElemsNum<PT>::value;
  if (n % packed_elems != 0) {
    packed_elems = 1;
  }

  size_t group_num = group_info.size() / 4;
  void* group_info_ptr = workspace;
  cudaMemcpyAsync(group_info_ptr, reinterpret_cast<const void*>(group_info.data()), group_info.size() * sizeof(size_t),
                  cudaMemcpyHostToDevice, stream);

  const size_t BLOCK_SIZE = 256;
  dim3 grid(m), block(BLOCK_SIZE);
  size_t for_range = ceil(static_cast<float>(n) / BLOCK_SIZE);
  if (packed_elems > 1) {
    DpMapCopyKernel<PT><<<grid, block, 0, stream>>>(reinterpret_cast<const PT*>(input), reinterpret_cast<PT*>(output),
                                                    static_cast<const size_t*>(group_info_ptr), group_num,
                                                    n / packed_elems, for_range / packed_elems);
  } else {
    DpMapCopyKernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
                                                   static_cast<const size_t*>(group_info_ptr), group_num, n, for_range);
  }
}

#define INSTANTIATE_MAP_COPY(T)                                                                             \
  template void InvokeDpMapCopy(const T* __restrict__ input, T* __restrict__ output,                        \
                                const std::vector<size_t>& group_info, size_t m, size_t n, void* workspace, \
                                cudaStream_t& stream);

INSTANTIATE_MAP_COPY(float);
INSTANTIATE_MAP_COPY(half);
INSTANTIATE_MAP_COPY(__nv_bfloat16);

#undef INSTANTIATE_MAP_COPY

}  // namespace nvidia
}  // namespace llm_kernels