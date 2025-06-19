# Copyright 2024 Tencent Inc.All rights reserved.

if(SM STREQUAL "90" OR SM STREQUAL "90a")
    add_definitions(-DENABLE_BLOCKWISE_GEMM=ON)
endif()

# Set kernels target
file(GLOB_RECURSE BLOCKWISE_GEMM_SRCS *.cu)
list(FILTER BLOCKWISE_GEMM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_blockwise_gemm STATIC ${BLOCKWISE_GEMM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_blockwise_gemm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_blockwise_gemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_blockwise_gemm PUBLIC -lcublas -lcudart -lcublasLt)

# For test
file(GLOB_RECURSE BLOCKWISE_GEMM_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_blockwise_gemm_test SRCS ${BLOCKWISE_GEMM_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_blockwise_gemm)
