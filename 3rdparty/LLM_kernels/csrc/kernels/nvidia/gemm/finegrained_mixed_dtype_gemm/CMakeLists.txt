# Copyright 2025 Tencent Inc.  All rights reserved.

if ((SM STREQUAL "89" OR SM STREQUAL "90" OR SM STREQUAL "90a") AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
    set(ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM 1)
else()
    set(ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM 0)
    message(WARNING "SM < 89 or CUDA version < 12.8, set ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0")
endif()

# set kernels target
file(GLOB_RECURSE FINEGRAINED_MIXED_DTYPE_GEMM_SRCS *.cu *.cpp)
list(FILTER FINEGRAINED_MIXED_DTYPE_GEMM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm STATIC ${FINEGRAINED_MIXED_DTYPE_GEMM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_definitions(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PUBLIC ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=${ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM})
target_compile_definitions(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PRIVATE NDEBUG)
target_compile_definitions(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PRIVATE ENABLE_BF16)
target_compile_definitions(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PRIVATE ENABLE_FP4)
target_link_libraries(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PUBLIC -lcublas -lcudart -lcublasLt
                      llm_kernels_nvidia_utils)
if (ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM)
    target_link_libraries(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PUBLIC llm_kernels_nvidia_kernel_others_trtllm_dev_thop)
endif()
target_link_libraries(llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm PRIVATE cutlass_c4x)