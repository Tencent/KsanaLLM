# Copyright 2025 Tencent Inc.  All rights reserved.

add_subdirectory(deepgemm)
add_subdirectory(finegrained_mixed_dtype_gemm)

# Create a unified interface target that includes sub-targets
add_library(llm_kernels_nvidia_kernel_gemm INTERFACE)
target_link_libraries(llm_kernels_nvidia_kernel_gemm INTERFACE
    llm_kernels_nvidia_kernel_gemm_deepgemm
    llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm
)