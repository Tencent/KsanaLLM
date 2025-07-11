/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/weight_only_batched_gemv/weight_only_gemv_wrapper.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "csrc/kernels/nvidia/weight_only_batched_gemv/kernelLauncher.h"

namespace llm_kernels {
namespace nvidia {

template <typename T, WeightType WT>
FpAIntBGroupCudaGemmWrapper<T, WT>::FpAIntBGroupCudaGemmWrapper() {
  arch = llm_kernels::utils::GetSMVersion();
}

template <typename T, WeightType WT>
bool FpAIntBGroupCudaGemmWrapper<T, WT>::IsSupport() {
  if constexpr (WT == WeightType::INT4 && std::is_same_v<T, half>) {
    return llm_kernels::nvidia::weight_only::is_supported(
        arch, llm_kernels::nvidia::weight_only::KernelType::FP16Int4Groupwise);
  } else {
    return false;
  }
}

template <typename T, WeightType WT>
void FpAIntBGroupCudaGemmWrapper<T, WT>::Gemm(void* output, const void* input, const void* weight, const void* scales,
                                              const void* zeros, size_t m, size_t n, size_t k, size_t groupsize,
                                              cudaStream_t stream) {
  if constexpr (WT == WeightType::INT4 && std::is_same_v<T, half>) {
    llm_kernels::nvidia::weight_only::Params params{reinterpret_cast<const void*>(input),
                                                    nullptr,
                                                    reinterpret_cast<const void*>(weight),
                                                    reinterpret_cast<const void*>(scales),
                                                    reinterpret_cast<const void*>(zeros),
                                                    nullptr,  // no bias
                                                    reinterpret_cast<void*>(output),
                                                    1.0f,
                                                    static_cast<int>(m),
                                                    static_cast<int>(n),
                                                    static_cast<int>(k),
                                                    static_cast<int>(groupsize),
                                                    weight_only::KernelType::FP16Int4Groupwise};
    llm_kernels::nvidia::weight_only::kernel_launcher(arch, params, stream);
  }
}

template class FpAIntBGroupCudaGemmWrapper<float, INT4>;
template class FpAIntBGroupCudaGemmWrapper<float, INT8>;

template class FpAIntBGroupCudaGemmWrapper<half, INT4>;
template class FpAIntBGroupCudaGemmWrapper<half, INT8>;

template class FpAIntBGroupCudaGemmWrapper<__nv_bfloat16, INT4>;
template class FpAIntBGroupCudaGemmWrapper<__nv_bfloat16, INT8>;

}  // namespace nvidia
}  // namespace llm_kernels