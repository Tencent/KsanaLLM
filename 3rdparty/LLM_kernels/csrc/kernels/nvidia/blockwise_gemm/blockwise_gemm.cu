/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.cuh"
#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void BlockwiseGemmKernel(void* a, float* a_scales, void* b, float* b_scales, void* out, int m, int k, int n,
                         cudaStream_t& stream, void* cutlass_buffer, size_t cutlass_buffer_size) {
#if defined(ENABLE_BLOCKWISE_GEMM)
  if constexpr (std::is_same<T, float>::value) {
    std::cerr << "BlockwiseGemmKernel do not support float type" << std::endl;
  } else if constexpr (std::is_same<T, half>::value) {
    ksana_llm::DispatchCutlassGemmBlockwiseSm90Fp8<cutlass::half_t>(a, a_scales, b, b_scales, out, cutlass_buffer,
                                                                    cutlass_buffer_size, m, k, n, stream);
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    ksana_llm::DispatchCutlassGemmBlockwiseSm90Fp8<cutlass::bfloat16_t>(a, a_scales, b, b_scales, out, cutlass_buffer,
                                                                        cutlass_buffer_size, m, k, n, stream);
  }
#else
  std::cerr << "SM version is lower than 90. skipping BlockwiseGemm Kernel. This may cause some prefision problems."
            << std::endl;
#endif
}
#define BLOCKWISE_GEMM_KERNEL(T)                                                                                    \
  template void BlockwiseGemmKernel<T>(void* a, float* a_scales, void* b, float* b_scales, void* out, int m, int k, \
                                       int n, cudaStream_t& stream, void* cutlass_buffer, size_t cutlass_buffer_size)
BLOCKWISE_GEMM_KERNEL(float);
BLOCKWISE_GEMM_KERNEL(half);
BLOCKWISE_GEMM_KERNEL(__nv_bfloat16);
#undef BLOCKWISE_GEMM_KERNEL

}  // namespace nvidia
}  // namespace llm_kernels
