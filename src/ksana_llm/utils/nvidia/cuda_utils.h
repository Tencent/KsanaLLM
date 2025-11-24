/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

static const char* GetErrorString(CUresult error) {
  const char* err_str;
  cuGetErrorString(error, &err_str);
  return err_str;
}

static const char* GetErrorString(cublasStatus_t error) {
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
  return "UNKNOWN";
}

static const char* GetErrorString(cudaError_t error) { return cudaGetErrorString(error); }

template <typename T>
void CheckCUDAError(T result, const char* func, const char* file, const int line) {
  if (result) {
    KLLM_LOG_ERROR << fmt::format("CUDA runtime error: {} {}:{}@{}", GetErrorString(result), file, line, func);
    abort();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
}

#define CUDA_CHECK(val) CheckCUDAError((val), #val, __FILE__, __LINE__)

#define CUDA_CHECK_LAST_ERROR(...)                            \
  do {                                                        \
    (__VA_ARGS__);                                            \
    cudaError_t result = cudaGetLastError();                  \
    CheckCUDAError(result, #__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define CUDA_CHECK_RETURN(status) \
  if (status != CUDA_SUCCESS) {   \
    return status;                \
  }

#define CU_CHECK(cmd)                                                                                 \
  do {                                                                                                \
    CUresult r = cmd;                                                                                 \
    if (r != CUDA_SUCCESS) {                                                                          \
      const char* error_string;                                                                       \
      cuGetErrorString(r, &error_string);                                                             \
      KLLM_LOG_ERROR << fmt::format("Failed: Cuda error:{} {}:{}", error_string, __FILE__, __LINE__); \
      exit(RetCode::RET_INVALID_ARGUMENT);                                                            \
    }                                                                                                 \
  } while (0)

template <typename Func>
float MeasureCudaExecutionTime(Func&& func, cudaStream_t stream, int warmups = 10, int iterations = 100) {
  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmups; ++i) {
    func();
  }

  CUDA_CHECK(cudaEventRecord(begin, stream));
  for (int i = 0; i < iterations; ++i) {
    func();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaEventSynchronize(end));

  float cost_time;
  CUDA_CHECK(cudaEventElapsedTime(&cost_time, begin, end));

  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));

  return cost_time / iterations;
}

// File descriptor for local IPC or fabric handle for remote access
union IpcMemHandle {
  uint64_t fd;
  CUmemFabricHandle fh;
};

struct NvlsHandle {
  // Unicast/Multicast object meta
  size_t size = 0;
  size_t granularity = 0;
  size_t mc_granularity = 0;
  size_t signal_pad_offset = 0;
  CUmemAllocationHandleType mem_handle_type;
  CUmemAccessDesc access_desc;
  CUmulticastObjectProp mcprop;
  // Unicast/Multicast device pointers
  uintptr_t uc_ptr = 0;
  uintptr_t mc_ptr = 0;
  CUdeviceptr uc_va;
  CUdeviceptr mc_va;
  std::vector<CUdeviceptr> peer_uc_vas;
  // Unicast/Multicast device allocation handles
  std::vector<CUmemGenericAllocationHandle> peer_uc_handles;
  CUmemGenericAllocationHandle uc_handle;
  CUmemGenericAllocationHandle mc_handle;
};

// Manages multicast memory for efficient inter-GPU communication
// with per-rank unicast pointers and a shared multicast pointer
class NvlsMcastMemory {
 public:
  static NvlsMcastMemory* GetInstance();

  void Initialize(const size_t device_count);

  std::vector<NvlsHandle*>& GetNvlsHandles();
  std::vector<IpcMemHandle>& GetIpcHandles();
  // Get the raw array of signal pad pointers to all ranks (including self)
  // Used for synchronization across devices
  std::vector<void**>& GetSignalPadsDev();

  // Allocate multicast memory required for multimem NVLinkSharp instructions
  void AllocMcastMemory(const size_t device_id, const size_t size);

  // Initialize the multicast memory used in multimem allreduce after allocation is finished
  void InitMcastMemory(const size_t device_id);

  // Free the multicast memory used in multimem allreduce
  void FreeMcastMemory(const size_t device_id);

 private:
  NvlsMcastMemory() = default;

  // Return CU_MEM_HANDLE_TYPE_FABRIC if GPU Fabric is available for fast inter-node memory sharing,
  // otherwise CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
  CUmemAllocationHandleType GetMemHandleType(const size_t device_id);

  // Used to store unicast/multicast info for multimem based communication
  std::vector<NvlsHandle*> nvls_handles_;
  std::vector<IpcMemHandle> ipc_handles_;
  std::vector<void**> signal_pads_dev_;

  static constexpr size_t kSignalPadSize = 2048;
};

}  // namespace ksana_llm
