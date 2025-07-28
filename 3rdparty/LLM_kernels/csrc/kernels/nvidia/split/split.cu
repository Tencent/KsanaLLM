/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/split/split.h"

#include <cuda_fp16.h>
#include <thrust/device_vector.h>

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;
namespace llm_kernels {
namespace nvidia {

template <typename T>
__device__ void SplitToMultiKernelImpl(const T* __restrict__ input, T** __restrict__ outputs,
                                       const int* __restrict__ col_offsets,  // [0, col1, col1+col2, ...]
                                       int cols, int num_outputs, int for_range) {
  size_t row_id = blockIdx.x;
  size_t col_start_id = threadIdx.x * for_range;
  if (col_start_id >= cols) {
    return;
  }
  size_t col_end_id = col_start_id + for_range;

  size_t start_output_id = 0;
  size_t end_output_id = num_outputs - 1;

  // assume num_outputs is very small, so linear search is fine
  for (size_t i = 0; i < num_outputs; ++i) {
    if (col_start_id < col_offsets[i + 1]) {
      start_output_id = i;
      break;
    }
  }
  for (size_t i = start_output_id; i < num_outputs; ++i) {
    if (col_end_id <= col_offsets[i + 1]) {
      end_output_id = i;
      break;
    }
  }

  if (start_output_id == end_output_id) {  // normal case
    size_t output_col_num = col_offsets[start_output_id + 1] - col_offsets[start_output_id];
    for (size_t offset = 0; offset < for_range && col_start_id + offset < cols; ++offset) {
      size_t col_id = col_start_id + offset;
      size_t input_idx = row_id * cols + col_id;
      size_t output_idx = row_id * output_col_num + (col_id - col_offsets[start_output_id]);
      outputs[start_output_id][output_idx] = input[input_idx];
    }
  } else {  // special case, split across multiple outputs
    for (size_t offset = 0; offset < for_range && col_start_id + offset < cols; ++offset) {
      size_t col_id = col_start_id + offset;
      size_t input_idx = row_id * cols + col_id;
      // Find the output index
      size_t output_idx = start_output_id;
      while (output_idx < end_output_id && col_id >= col_offsets[output_idx + 1]) {
        ++output_idx;
      }
      size_t local_col_id = col_id - col_offsets[output_idx];
      outputs[output_idx][row_id * (col_offsets[output_idx + 1] - col_offsets[output_idx]) + local_col_id] =
          input[input_idx];
    }
  }
}

template <typename T>
__global__ void SplitToMultiKernel(const T* __restrict__ input, T** __restrict__ outputs,
                                   const int* __restrict__ col_offsets,  // [0, col1, col1+col2, ...]
                                   int cols, int num_outputs, int for_range) {
  SplitToMultiKernelImpl(input, outputs, col_offsets, cols, num_outputs, for_range);
}

template <typename T>
__global__ void SplitTo3Kernel(const T* __restrict__ input, T* __restrict__ output_a, T* __restrict__ output_b,
                               T* __restrict__ output_c, const int col_offset_a, const int col_offset_b,
                               const int col_offset_c, int cols, int num_outputs, int for_range) {
  T* outputs[3] = {output_a, output_b, output_c};
  int col_offsets[4] = {0, col_offset_a, col_offset_b, col_offset_c};
  SplitToMultiKernelImpl(input, outputs, col_offsets, cols, num_outputs, for_range);
}

template <typename T>
__global__ void SplitTo2Kernel(const T* __restrict__ input, T* __restrict__ output_a, T* __restrict__ output_b,
                               const int col_sep, int cols, int for_range) {
  size_t row_id = blockIdx.x;
  size_t col_start_id = threadIdx.x;
  for (size_t i = 0; i < for_range; i++) {
    size_t col_id = col_start_id + i * blockDim.x;
    if (col_id >= cols) {
      return;
    }
    size_t input_idx = row_id * cols + col_id;
    T* output = (col_id < col_sep) ? output_a : output_b;
    size_t local_col_id = (col_id < col_sep) ? col_id : (col_id - col_sep);
    size_t output_idx = ((col_id < col_sep) ? col_sep : cols - col_sep) * row_id + local_col_id;
    output[output_idx] = input[input_idx];
  }
}

template <typename T>
void SplitToMultiMemcpy2D(const T* __restrict__ input, const std::vector<T*>& output_ptrs,
                          std::vector<int>& col_offsets,  // [0, col1, col1+col2, ...]
                          int rows, int cols, int num_outputs, cudaStream_t& stream) {
  for (int i = 0; i < num_outputs; ++i) {
    int col_start = col_offsets[i];
    int col_width = col_offsets[i + 1] - col_offsets[i];

    const T* src_ptr = input + col_start;
    cudaMemcpy2DAsync(output_ptrs[i], col_width * sizeof(T), src_ptr, cols * sizeof(T), col_width * sizeof(T), rows,
                      cudaMemcpyDeviceToDevice, stream);
  }
}

template <typename T>
void InvokeSplit(const T* __restrict__ input, const std::vector<T*>& output_ptrs,
                 std::vector<int> col_offsets,  // [0, col1, col1+col2, ...]
                 int rows, int cols, int num_outputs, cudaStream_t& stream) {
  if (num_outputs > 3) {
    // num_outputs > 3 use memcpy2d which needn't packed
    SplitToMultiMemcpy2D<T>(reinterpret_cast<const T*>(input), output_ptrs, col_offsets, rows, cols, num_outputs,
                            stream);
    return;
  }
  using PT = typename PackTypeAlign<T>::type;
  int32_t packed_elems = ElemsNum<PT>::value;
  for (int offset : col_offsets) {
    if (offset % packed_elems != 0) {
      packed_elems = 1;  // fallback to non-packed if any offset is not aligned
    }
  }
  for (int i = 0; i < num_outputs + 1; ++i) {
    col_offsets[i] /= packed_elems;
  }
  const size_t BLOCK_SIZE = 256;
  dim3 grid(rows), block(BLOCK_SIZE);
  size_t for_range = ceil(static_cast<float>(cols) / packed_elems / BLOCK_SIZE);

  if (num_outputs == 2) {
    if (packed_elems == 1) {  // not using PT
      SplitTo2Kernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input), output_ptrs[0], output_ptrs[1],
                                                    col_offsets[1], cols, for_range);
    } else {
      SplitTo2Kernel<PT><<<grid, block, 0, stream>>>(
          reinterpret_cast<const PT*>(input), reinterpret_cast<PT*>(output_ptrs[0]),
          reinterpret_cast<PT*>(output_ptrs[1]), col_offsets[1], cols / packed_elems, for_range);
    }
  } else if (num_outputs == 3) {
    if (packed_elems == 1) {  // not using PT
      SplitTo3Kernel<T><<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input), output_ptrs[0], output_ptrs[1],
                                                    output_ptrs[2], col_offsets[1], col_offsets[2], col_offsets[3],
                                                    cols, num_outputs, for_range);
    } else {
      SplitTo3Kernel<PT><<<grid, block, 0, stream>>>(
          reinterpret_cast<const PT*>(input), reinterpret_cast<PT*>(output_ptrs[0]),
          reinterpret_cast<PT*>(output_ptrs[1]), reinterpret_cast<PT*>(output_ptrs[2]), col_offsets[1], col_offsets[2],
          col_offsets[3], cols / packed_elems, num_outputs, for_range);
    }
  }
}

#define INSTANTIATE_SPLIT(T)                                                                 \
  template void InvokeSplit(const T* __restrict__ input, const std::vector<T*>& output_ptrs, \
                            std::vector<int> col_offsets, int rows, int cols, int num_outputs, cudaStream_t& stream);

INSTANTIATE_SPLIT(float);
INSTANTIATE_SPLIT(half);
INSTANTIATE_SPLIT(__nv_bfloat16);

#undef INSTANTIATE_SPLIT

}  // namespace nvidia
}  // namespace llm_kernels