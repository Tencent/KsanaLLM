/*
 * Copyright 2025 vLLM Team
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
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/tree/v0.8.2/csrc/cutlass_extensions
 */

#pragma once

#include "cutlass/cutlass.h"
#include <climits>
#include "cuda_runtime.h"
#include <iostream>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    KLLM_KERNEL_CHECK_WITH_INFO(error == cutlass::Status::kSuccess, cutlassGetStatusString(error));     \
  }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                        \
  {                                                               \
    cudaError_t error = status;                                   \
    KLLM_KERNEL_CHECK_WITH_INFO(error == cudaSuccess, cudaGetErrorString(error)); \
  }

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  return max_shared_mem_per_block_opt_in;
}

int32_t get_sm_version_num();

/**
 * A wrapper for a kernel that is used to guard against compilation on
 * architectures that will never use the kernel. The purpose of this is to
 * reduce the size of the compiled binary.
 * __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
 * into code that will be executed on the device where it is defined.
 */
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};