/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/machete/machete_wrapper.h"

#include "csrc/kernels/nvidia/machete/machete_mm_launcher.cuh"
#include "csrc/kernels/nvidia/machete/machete_prepack_launcher.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace machete {

template <typename T>
__global__ void unpackInt4x2Kernel(const uint8_t* packed, T* unpacked, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    unpacked[2 * idx] = static_cast<T>(static_cast<int8_t>(packed[idx] % 16) - 8);
    unpacked[2 * idx + 1] = static_cast<T>(static_cast<int8_t>(packed[idx] / 16) - 8);
  }
}

template <typename T>
void unpackInt4x2(cudaStream_t stream, const uint8_t* packed, T* unpacked, const size_t len) {
  int threadsPerBlock = 128;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
  unpackInt4x2Kernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(packed, unpacked, len);
}

__global__ void unpackScaleKernel(const float* packed, float* unpacked, size_t packedRows, size_t n, size_t groupsize) {
  size_t i = blockIdx.x;
  size_t j = blockIdx.y;
  size_t g = threadIdx.x;
  if (i < packedRows && j < n && g < groupsize) {
    size_t unpackedIndex = (i * groupsize + g) * n + j;
    size_t packedIndex = i * n + j;
    unpacked[unpackedIndex] = packed[packedIndex];
  }
}

void unpackScale(cudaStream_t stream, const float* packed, float* unpacked, size_t k, size_t n, size_t groupsize) {
  size_t packedRows = k / groupsize;
  dim3 blockSize(groupsize);
  dim3 gridSize(packedRows, n);
  unpackScaleKernel<<<gridSize, blockSize, 0, stream>>>(packed, unpacked, packedRows, n, groupsize);
}

__global__ void elementwiseMulKernel(const float* A, const float* B, float* C, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    C[idx] = A[idx] * B[idx];
  }
}

void elementwiseMul(cudaStream_t stream, const float* A, const float* B, float* C, size_t len) {
  size_t blockSize = 128;
  size_t gridSize = (len + blockSize - 1) / blockSize;
  elementwiseMulKernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, len);
}

std::vector<std::string> machete_supported_schedules(vllm_dtype::ScalarType a_type, vllm_dtype::ScalarType b_type,
                                                     std::optional<vllm_dtype::ScalarType> maybe_group_scales_type,
                                                     std::optional<vllm_dtype::ScalarType> maybe_group_zeros_type) {
#if defined(ENABLE_MACHETE)
  return supported_schedules_dispatch({.a_type = a_type,
                                       .b_type = b_type,
                                       .maybe_group_scales_type = maybe_group_scales_type,
                                       .maybe_group_zeros_type = maybe_group_zeros_type,
                                       .maybe_channel_scales_type = std::nullopt,
                                       .maybe_token_scales_type = std::nullopt,
                                       .maybe_out_type = std::nullopt});
#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping Machete Kernel.");
#endif
}

void machete_gemm(int64_t& workspace_size, void* workspace, cudaStream_t stream, int M, int N, int K, const void* Aptr,
                  const void* Bptr, void* Dptr, vllm_dtype::ScalarType const& a_type,
                  vllm_dtype::ScalarType const& b_type, std::optional<void*> const& maybe_group_scales_ptr,
                  std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                  std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type,
                  std::optional<void*> const& maybe_group_zeros_ptr,
                  std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                  std::optional<vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                  std::optional<int64_t> maybe_group_size, std::optional<std::string> maybe_schedule) {
#if defined(ENABLE_MACHETE)
  mm_dispatch({.workspace_size = workspace_size,
               .workspace = workspace,
               .stream = stream,
               .M = M,
               .N = N,
               .K = K,
               .Aptr = Aptr,
               .A_shape = {static_cast<size_t>(M), static_cast<size_t>(K)},
               .Bptr = Bptr,
               .Dptr = Dptr,
               .D_shape = {static_cast<size_t>(M), static_cast<size_t>(N)},
               .a_type = a_type,
               .b_type = b_type,
               .maybe_out_type = std::nullopt,
               .maybe_group_scales_ptr = maybe_group_scales_ptr,
               .maybe_group_scales_shape = maybe_group_scales_shape,
               .maybe_group_scales_type = maybe_group_scales_type,
               .maybe_group_zeros_ptr = maybe_group_zeros_ptr,
               .maybe_group_zeros_shape = maybe_group_zeros_shape,
               .maybe_group_zeros_type = maybe_group_zeros_type,
               .maybe_group_size = maybe_group_size,
               .maybe_channel_scales_ptr = std::nullopt,
               .maybe_channel_scales_numel = std::nullopt,
               .maybe_channel_scales_type = std::nullopt,
               .maybe_token_scales_ptr = std::nullopt,
               .maybe_token_scales_numel = std::nullopt,
               .maybe_token_scales_type = std::nullopt,
               .maybe_schedule = maybe_schedule});
#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping Machete Kernel.");
#endif
}

void machete_prepack_weight(const void* B_ptr, const std::vector<size_t>& B_shape, void* out_ptr,
                            vllm_dtype::ScalarType const& a_type, vllm_dtype::ScalarType const& b_type,
                            std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type, cudaStream_t stream) {
#if defined(ENABLE_MACHETE)
  prepack_B_dispatch({.B_ptr = B_ptr,
                      .B_shape = B_shape,
                      .a_type = a_type,
                      .b_type = b_type,
                      .maybe_group_scales_type = maybe_group_scales_type},
                     out_ptr, stream);
#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping Machete Kernel.");
#endif
}

std::string machete_best_schedule(size_t warmup_iters, size_t record_iters, void* workspace, cudaStream_t stream, int M,
                                  int N, int K, const void* Aptr, const void* Bptr, void* Dptr,
                                  vllm_dtype::ScalarType const& a_type, vllm_dtype::ScalarType const& b_type,
                                  std::optional<void*> const& maybe_group_scales_ptr,
                                  std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                                  std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type,
                                  std::optional<void*> const& maybe_group_zeros_ptr,
                                  std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                                  std::optional<vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                                  std::optional<int64_t> maybe_group_size) {
#if defined(ENABLE_MACHETE)
  std::string best_schedule_string;
  float best_schedule_time = std::numeric_limits<float>::max();
  std::vector<std::string> schedules =
      machete_supported_schedules(a_type, b_type, maybe_group_scales_type, maybe_group_zeros_type);

  for (size_t schedule_idx = 0; schedule_idx < schedules.size(); schedule_idx++) {
    std::string curr_schedule = schedules[schedule_idx];

    auto cuda_run = [&]() {
      int64_t current_workspace_size = 0;
      machete_gemm(current_workspace_size, workspace, stream, M, N, K, Aptr, Bptr, Dptr, a_type, b_type,
                   maybe_group_scales_ptr, maybe_group_scales_shape, maybe_group_scales_type, maybe_group_zeros_ptr,
                   maybe_group_zeros_shape, maybe_group_zeros_type, maybe_group_size, curr_schedule);
    };
    float curr_time = MeasureCudaExecutionTime(cuda_run, stream, warmup_iters, record_iters);

    if (curr_time < best_schedule_time) {
      best_schedule_time = curr_time;
      best_schedule_string = curr_schedule;
    }
  }

  return best_schedule_string;
#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping Machete Kernel.");
#endif
}

template void unpackInt4x2<float>(cudaStream_t stream, const uint8_t* packed, float* unpacked, const size_t len);
template void unpackInt4x2<int8_t>(cudaStream_t stream, const uint8_t* packed, int8_t* unpacked, const size_t len);

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
