/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

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

}  // namespace ksana_llm
