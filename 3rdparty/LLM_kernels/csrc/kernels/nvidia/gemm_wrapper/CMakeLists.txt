# Copyright 2024 Tencent Inc.  All rights reserved.

# set masked multihead attention kernels target
file(GLOB_RECURSE GEMM_WRAPPER_SRCS *.cu)
list(FILTER GEMM_WRAPPER_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_gemm_wrapper STATIC ${GEMM_WRAPPER_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_gemm_wrapper PUBLIC -lcublas -lcudart -lcublasLt yaml-cpp)
target_link_libraries(llm_kernels_nvidia_kernel_gemm_wrapper PRIVATE cutlass_c3x)
set_property(TARGET llm_kernels_nvidia_kernel_gemm_wrapper PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_gemm_wrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
file(GLOB_RECURSE GEMM_WRAPPER_TEST_SRCS *test.cu)
message(STATUS "GEMM_WRAPPER_TEST_SRCS ${GEMM_WRAPPER_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_gemm_wrapper_test SRCS ${GEMM_WRAPPER_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils    
    llm_kernels_nvidia_kernel_gemm_wrapper ksana_llm_yaml)
