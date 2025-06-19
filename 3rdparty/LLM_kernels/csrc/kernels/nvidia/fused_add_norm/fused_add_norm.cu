/* Copyright 2025 Tencent Inc.  All rights reserved.
==============================================================================*/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_BFLOAT16
#  include <cuda_bf16.h>
#endif

#include <flashinfer/norm.cuh>

#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeFusedAddRMSNorm(void* input, void* residual, void* weight, double eps, bool enable_pdl, uint32_t m,
                           uint32_t n, cudaStream_t stream) {
// Step 1: residual[i] += input[i]
// Step 2: input[i] = (residual[i] / RMS(residual)) * weight[i]
#if defined(ENABLE_FLASHINFER)
  flashinfer::norm::FusedAddRMSNorm(static_cast<T*>(input), static_cast<T*>(residual), static_cast<T*>(weight), m, n,
                                    /*stride_input*/ n, /*stride_residual*/ n, eps, enable_pdl, stream);
#else
  std::cerr << "ENABLE_FLASHINFER is not defined. skipping invoke flashinfer Kernel." << std::endl;
#endif
}
#define FUSED_ADD_RMS_NORM(T)                                                                                    \
  template void InvokeFusedAddRMSNorm<T>(void* input, void* residual, void* weight, double eps, bool enable_pdl, \
                                         uint32_t m, uint32_t n, cudaStream_t stream)

FUSED_ADD_RMS_NORM(float);
FUSED_ADD_RMS_NORM(half);

#ifdef ENABLE_BFLOAT16
FUSED_ADD_RMS_NORM(__nv_bfloat16);
#endif
#undef FUSED_ADD_RMS_NORM

}  // namespace nvidia
}  // namespace llm_kernels