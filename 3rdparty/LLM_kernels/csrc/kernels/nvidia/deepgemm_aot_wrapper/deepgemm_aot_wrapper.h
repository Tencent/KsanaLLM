/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"

namespace llm_kernels {
namespace nvidia {

class DeepGEMMAOTWrapper {
 public:
  DeepGEMMAOTWrapper(size_t m, size_t n, size_t k, bool need_generate_kernel, size_t tuner_device_id = 0);
  DeepGEMMAOTWrapper(const DeepGEMMOpConfig& op_config);
  ~DeepGEMMAOTWrapper();
  cudaError_t Forward(void* raw_lhs, void* raw_lhs_scales, void* raw_rhs, void* raw_rhs_scales, void* raw_out, int m,
                      void* raw_stream);

 private:
  bool IsKernelExist(const std::string& directory);
  void LoadFromYaml(const std::string& config_file);
  void ThrowError(const std::string& error_msg);
  std::string InitCacheDir();
  void LoadFuncFromSO();

 private:
  using LaunchFunc = void (*)(void*, void*, void*, void*, void*, int, void*, int, int, int&);
  void* handle_ = nullptr;  // handle of dynamic library path
  DeepGEMMOpConfig op_config_;
  LaunchFunc launch_func_ = nullptr;  // function pointer for launch function
};

}  // namespace nvidia
}  // namespace llm_kernels