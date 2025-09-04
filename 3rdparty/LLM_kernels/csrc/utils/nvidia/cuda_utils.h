/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nvml.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include "csrc/utils/nvidia/assert.h"

#ifdef ENABLE_FP8
#  include "cuda_fp8_utils.h"
#endif

namespace llm_kernels {
namespace utils {

constexpr int32_t NVIDIA_VOLTA_GPU_COMPUTE_CAPABILITY = 70;
constexpr int32_t NVIDIA_AGX_XAVIER_GPU_COMPUTE_CAPABILITY = 72;
constexpr int32_t NVIDIA_TURING_GPU_COMPUTE_CAPABILITY = 75;
constexpr int32_t NVIDIA_AMPERE_GPU_COMPUTE_CAPABILITY = 80;
constexpr int32_t NVIDIA_HOPPER_GPU_COMPUTE_CAPABILITY = 90;

constexpr int32_t DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM = 65536;
constexpr int32_t DEFAULT_CUDA_BLOCK_THREADS_NUM = 512;
constexpr int32_t DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM = 256;
constexpr int32_t DEFAULT_CUDA_MAX_BLOCKS_NUM = 8192;
constexpr int32_t DEFAULT_CUDA_WARP_SIZE = 32;
constexpr int32_t DEFAULT_CUDA_HALF_WARP_SIZE = 16;
constexpr int32_t DEFAULT_CUDA_QUARTER_WARP_SIZE = 8;
constexpr int32_t DEFAULT_CUDA_ONE_EIGHTH_WARP_SIZE = 4;
constexpr int32_t DEFAULT_CUDA_ONE_SIXTEENTH_WARP_SIZE = 2;
constexpr int32_t DEFAULT_CUDA_ONE_THIRTY_TWO_WARP_SIZE = 1;

static const char* GetErrorCode(nvmlReturn_t error) { return nvmlErrorString(error); }

static const char* GetErrorCode(cudaError_t error) { return cudaGetErrorString(error); }

static inline const char* GetErrorCode(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void CheckNvidiaCUDAError(T result, const char* func, const char* file, const int32_t line) {
  if (result) {
    throw std::runtime_error(std::string("[LLMKernels] CUDA runtime error: ") + (GetErrorCode(result)) + " " + file +
                             ":" + std::to_string(line) + "@" + func + " \n");
  }
}

#define CHECK_NVIDIA_CUDA_ERROR(val) CheckNvidiaCUDAError((val), #val, __FILE__, __LINE__)
// refer to
// https://github.com/NVIDIA/TensorRT-LLM/blame/ab49b93718b906030bcec0c817b10ebb373d4179/cpp/include/tensorrt_llm/common/cudaUtils.h
inline std::optional<bool> IsCudaLaunchBlocking() {
  static bool first_call = true;
  static std::optional<bool> result = std::nullopt;

  if (first_call) {
    char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (env != nullptr && std::string(env) == "1") {
      result = true;
    } else if (env != nullptr && std::string(env) == "0") {
      result = false;
    }
    first_call = false;
  }
  return result;
}

inline void SyncAndCheck(char const* const file, int const line) {
  auto const cuda_launch_blocking = IsCudaLaunchBlocking();
  bool const check_error = cuda_launch_blocking.value_or(false);

  if (check_error) {
    cudaError_t result = cudaDeviceSynchronize();
    CheckNvidiaCUDAError(result, "cudaDeviceSynchronize", file, line);
  }
}

#define sync_check_cuda_error() llm_kernels::utils::SyncAndCheck(__FILE__, __LINE__)

#define RETURN_NVIDIA_CUBLAS_ERROR(val) \
  if ((val)) {                          \
    return val;                         \
  }

template <typename T>
void RandomGPUBuffer(T* data_ptr, size_t n_elems, const float max_val = 1.0f, const float min_val = -1.0f);

template <typename T_INPUT, typename T_STEP>
void InvokeRange(T_INPUT* output, T_INPUT start, int32_t nstep, T_STEP step, cudaStream_t stream);

template <typename T>
void ResetGPUBufferWithStep(T* data_ptr, size_t n_elems, const float max_val = 1.0f, const float min_val = -1.0f,
                            const float val_step = 0.000001f);

typedef struct __align__(4) { half x, y, z, w; }
half4;

inline int32_t div_up(int32_t a, int32_t n) { return (a + n - 1) / n; }

template <typename T>
struct PackTypeAlign;
template <>
struct PackTypeAlign<float> {
  // we don't need to pack float by default
  using type = float;
};
template <>
struct PackTypeAlign<half> {
  using type = half2;
};

template <>
struct PackTypeAlign<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

typedef struct __CUDA_ALIGN__(4) {
  __nv_bfloat16 array[2];
} __nv_bfloat16_2;

typedef struct __CUDA_ALIGN__(8) {
  __nv_bfloat162 x, y;
} __nv_bfloat162_2_xy;

typedef struct __CUDA_ALIGN__(8) {
  // __nv_bfloat16 array[4];
  __nv_bfloat16 x, y, z, w;
} __nv_bfloat164;

typedef struct __CUDA_ALIGN__(8) {
  __nv_bfloat162 array[2];
} __nv_bfloat162_2;

typedef struct __CUDA_ALIGN__(16) {
  __nv_bfloat16 array[8];
} __nv_bfloat168;

typedef struct __CUDA_ALIGN__(16) {
  __nv_bfloat162 array[4];
} __nv_bfloat162_4;

typedef struct __CUDA_ALIGN__(32) {
  __nv_bfloat16 array[16];
} __nv_bfloat1616;

template <typename T>
struct ElemsNum;
template <>
struct ElemsNum<float> {
  static constexpr int32_t value = 1;
};
template <>
struct ElemsNum<float2> {
  static constexpr int32_t value = 2;
};
template <>
struct ElemsNum<float4> {
  static constexpr int32_t value = 4;
};
template <>
struct ElemsNum<half> {
  static constexpr int32_t value = 1;
};
template <>
struct ElemsNum<half2> {
  static constexpr int32_t value = 2;
};
template <>
struct ElemsNum<half4> {
  static constexpr int32_t value = 4;
};
template <>
struct ElemsNum<__nv_bfloat16> {
  static constexpr int32_t value = 1;
};
template <>
struct ElemsNum<__nv_bfloat162> {
  static constexpr int32_t value = 2;
};
template <>
struct ElemsNum<__nv_bfloat164> {
  static constexpr int32_t value = 4;
};
template <>
struct ElemsNum<__nv_bfloat168> {
  static constexpr int32_t value = 8;
};

template <typename T, int32_t PACK_SIZE>
struct PackType;
template <typename T>
struct PackType<T, 1> {
  using type = T;
};

template <>
struct PackType<half, 2> {
  using type = half2;
};
template <>
struct PackType<half, 4> {
  using type = half4;
};
template <>
struct PackType<float, 2> {
  using type = float2;
};
template <>
struct PackType<float, 4> {
  using type = float4;
};
template <>
struct PackType<int8_t, 2> {
  using type = int16_t;
};
template <>
struct PackType<int32_t, 2> {
  using type = int2;
};

template <>
struct PackType<half2, 1> {
  using type = half;
};

template <>
struct PackType<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};
template <>
struct PackType<__nv_bfloat16, 4> {
  using type = __nv_bfloat164;
};
template <>
struct PackType<__nv_bfloat16, 8> {
  using type = __nv_bfloat168;
};

template <>
struct PackType<__nv_bfloat162, 1> {
  using type = __nv_bfloat16;
};

#ifdef ENABLE_FP8
template <>
struct PackType<__nv_fp8_e4m3, 2> {
  using type = __nv_fp8_2_e4m3;
};

template <>
struct PackType<__nv_fp8_e4m3, 4> {
  using type = __nv_fp8_4_e4m3;
};

template <>
struct PackType<__nv_fp8_e4m3, 8> {
  using type = __nv_fp8_8_e4m3;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator*(float2 a, float b) { return make_float2(a.x * b, a.y * b); }

// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); i += step)

#define CUDA_1D_KERNEL_LOOP_T(type, i, n) \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); i += step)

inline int64_t BlocksNum4ThreadsNum(const int64_t thread_num) {
  return std::min((thread_num + DEFAULT_CUDA_BLOCK_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_THREADS_NUM,
                  static_cast<int64_t>(DEFAULT_CUDA_MAX_BLOCKS_NUM));
}

inline int32_t GetSMVersion() {
  int32_t device{-1};
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device));
  int32_t sm_major{0};
  int32_t sm_minor{0};
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline int GetSMCount() {
  static int sm_count{-1};
  if (sm_count == -1) {
    int device_id;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device_id));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

inline int GetPerBlockRegisterCount() {
  static int regs_per_block{-1};
  if (regs_per_block == -1) {
    int device_id;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device_id));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&regs_per_block, cudaDevAttrMaxRegistersPerBlock, device_id));
  }
  return regs_per_block;
}

inline int getMaxSharedMemoryPerBlockOptin() {
  int device_id;
  int max_shared_memory_per_block;
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device_id));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaDeviceGetAttribute(&max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
  return max_shared_memory_per_block;
}

inline int getMaxThreadPerBlock() {
  int device_id;
  int max_threads_per_block;
  CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device_id));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
  return max_threads_per_block;
}

/// Get the memory info
/// \return The free and total amount of memory in bytes
inline std::tuple<size_t, size_t> getDeviceMemoryInfo(bool const useUvm) {
  size_t free, total;
  CHECK_NVIDIA_CUDA_ERROR(cudaMemGetInfo(&free, &total));
  return {free, total};
}

inline int GetDeviceCount() {
  static int device_count{-1};
  if (device_count == -1) {
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  }
  return device_count;
}

uint32_t GetNvLinkVersion(uint32_t device_id, uint32_t link_idx);

template <typename Func>
float measureCPUExecutionTime(Func&& func, int warmups = 10, int iterations = 100) {
  for (int i = 0; i < warmups; ++i) {
    func();
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    func();
  }
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / iterations;
}

template <typename Func>
float MeasureCudaExecutionTime(Func&& func, cudaStream_t stream, int warmups = 10, int iterations = 100) {
  cudaEvent_t begin, end;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&end));

  for (int i = 0; i < warmups; ++i) {
    func();
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(begin, stream));
  for (int i = 0; i < iterations; ++i) {
    func();
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(end, stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(end));

  float cost_time;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&cost_time, begin, end));

  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(begin));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(end));

  return cost_time / iterations;
}

// Get the next power of 2 of a number
inline uint32_t NextPow2(uint32_t x) noexcept { return x <= 1u ? 1u : 1u << (32 - __builtin_clz(x - 1)); }

inline bool GetEnablePDL() {
  // We enable PDL (programmatic dependent launch) by default on hopper and later arch
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization
  static bool enable_pdl = GetSMVersion() >= 90 && std::getenv("KLLM_DISABLE_PDL") == nullptr;
  return enable_pdl;
}

inline bool EnableGpuP2PAccess(int device_count) {
  constexpr uint32_t kReserveP2PFlag = 0;
  for (int src_rank = 0; src_rank < device_count; src_rank++) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(src_rank));
    for (int dst_rank = 0; dst_rank < device_count; dst_rank++) {
      if (src_rank == dst_rank) {
        continue;
      }
      int can_cuda_enable_p2p = 0;
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_cuda_enable_p2p, src_rank, dst_rank));
      if (can_cuda_enable_p2p == 0) {
        return false;
      }
      cudaError_t err = cudaDeviceEnablePeerAccess(dst_rank, kReserveP2PFlag);
      if (err != cudaErrorPeerAccessAlreadyEnabled) {
        CHECK_NVIDIA_CUDA_ERROR(err);
      }
    }
  }
  return true;
}

}  // namespace utils
}  // namespace llm_kernels
