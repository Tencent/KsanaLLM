/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <tuple>

namespace llm_kernels {
namespace nvidia {

class DeepGEMMWrapper {
 public:
  DeepGEMMWrapper(int id);

  void BuildGemmKernel(int m, int n, int k);

  void Gemm(void* x_fp8, void* x_scales, void* y_fp8, void* y_scales, void* out, int m, int n, int k,
            cudaStream_t stream);

  void BuildGemmSwapABKernel(int m, int n, int k);

  void GemmSwapAB(void* x_fp8, void* x_scales, void* y_fp8, void* y_scales, void* out, int m, int n, int k,
                  cudaStream_t stream);

 private:
  int num_device_sms_;
  int id_{0};
};

}  // namespace nvidia
}  // namespace llm_kernels