/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "cuda_utils.h"

namespace llm_kernels {
namespace utils {

template <typename T>
__global__ void RunCUDARandomUniformKernel(T* buffer, const size_t size, const int32_t seq_offset, const float max_val,
                                           const float min_val) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t local_state;
  constexpr uint64_t SEED = 1337ul;
  constexpr float EPS = 1e-10;
  curand_init(SEED, idx + seq_offset, 0, &local_state);
  for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
    if constexpr (std::is_integral_v<T>) {
      // ceil((min_val, max_val + 1] - EPS) -> [min_val, max_val]
      buffer[index] = (T)(curand_uniform(&local_state) * (max_val - min_val + 1) + min_val - EPS);
    } else {
      // TODO(yfnjin): consider the constraints of `min_val` and `max_val` for float type
      // NOTE(karlluo): some cuda's kernel has not static_cast for half
      if (max_val == min_val) {
        buffer[index] = (T)(max_val);
      } else {
        buffer[index] = (T)(curand_uniform(&local_state) * 0.2f - 0.1f);
      }
    }
  }
}

template <typename T>
void RandomGPUBuffer(T* data_ptr, size_t n_elems, const float max_val, const float min_val) {
  static int32_t seq_offset = 0;
  constexpr int32_t random_tile_size = DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM;
  RunCUDARandomUniformKernel<T>
      <<<random_tile_size, random_tile_size>>>(data_ptr, n_elems, seq_offset, max_val, min_val);
}

template void RandomGPUBuffer(float* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(half* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(__nv_bfloat16* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(int32_t* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(bool* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(char* data_ptr, const size_t n_elems, const float max_val, const float min_val);
#ifdef ENABLE_FP8
template void RandomGPUBuffer(__nv_fp8_e4m3* data_ptr, const size_t n_elems, const float max_val, const float min_val);
#endif
template void RandomGPUBuffer(long* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(uint16_t* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(unsigned long* data_ptr, const size_t n_elems, const float max_val, const float min_val);
template void RandomGPUBuffer(uint32_t* data_ptr, const size_t n_elems, const float max_val, const float min_val);

template <typename T>
__global__ void ResetGPUBufferWithStepKernel(T* buffer, const size_t size, const float max_val, const float min_val,
                                             const float val_step) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
    float val = fmodf(index * val_step, (max_val - min_val)) + min_val;
    buffer[index] = (T)(val);
  }
}

template <typename T>
void ResetGPUBufferWithStep(T* data_ptr, size_t n_elems, const float max_val, const float min_val,
                            const float val_step) {
  constexpr int32_t random_tile_size = DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM;
  ResetGPUBufferWithStepKernel<T>
      <<<random_tile_size, random_tile_size>>>(data_ptr, n_elems, max_val, min_val, val_step);
}

template void ResetGPUBufferWithStep(float* data_ptr, const size_t n_elems, const float max_val, const float min_val,
                                     const float step);
template void ResetGPUBufferWithStep(half* data_ptr, const size_t n_elems, const float max_val, const float min_val,
                                     const float step);
template void ResetGPUBufferWithStep(__nv_bfloat16* data_ptr, const size_t n_elems, const float max_val,
                                     const float min_val, const float step);

template <typename T_INPUT, typename T_STEP>
__global__ void InvokeRangeKernel(T_INPUT* output, T_INPUT start, int32_t nstep, T_STEP step) {
  int32_t istep = blockIdx.x * blockDim.x + threadIdx.x;
  if (istep < nstep) {
    output[istep] = start + istep * step;
  }
}

template <typename T_INPUT, typename T_STEP>
void InvokeRange(T_INPUT* output, T_INPUT start, int32_t nstep, T_STEP step, cudaStream_t stream) {
  dim3 grid((nstep + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM);
  dim3 block(DEFAULT_CUDA_BLOCK_THREADS_NUM);
  InvokeRangeKernel<<<grid, block, 0, stream>>>(output, start, nstep, step);
}

template void InvokeRange(uint16_t** output, uint16_t* start, int32_t nstep, int32_t step, cudaStream_t stream);
template void InvokeRange(int32_t* output, int32_t start, int32_t nstep, int32_t step, cudaStream_t stream);

uint32_t GetNvLinkVersion(uint32_t device_id, uint32_t link_idx) {
  uint32_t version = 0;
  CHECK_NVIDIA_CUDA_ERROR(nvmlInit());
  nvmlDevice_t device;
  CHECK_NVIDIA_CUDA_ERROR(nvmlDeviceGetHandleByIndex(device_id, &device));
  if (nvmlDeviceGetNvLinkVersion(device, link_idx, &version) == NVML_ERROR_NOT_SUPPORTED) {
    return 0;
  }
  CHECK_NVIDIA_CUDA_ERROR(nvmlShutdown());
  return version;
}

}  // namespace utils
}  // namespace llm_kernels
