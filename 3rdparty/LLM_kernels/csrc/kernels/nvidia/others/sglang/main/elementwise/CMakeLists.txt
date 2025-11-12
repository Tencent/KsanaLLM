# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ELEMENTWISE_SRCS *.cu)
list(FILTER ELEMENTWISE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_sglang_elementwise STATIC ${ELEMENTWISE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_sglang_elementwise PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_sglang_elementwise PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_sglang_elementwise PUBLIC -lcublas -lcudart -lcublasLt
    llm_kernels_nvidia_utils)

file(GLOB_RECURSE ELEMENTWISE_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_sglang_elementwise_test SRCS ${ELEMENTWISE_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_sglang_elementwise
    llm_kernels_nvidia_kernel_concat
    llm_kernels_nvidia_kernel_expand)
