# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE CONCAT_SRCS *.cu)
list(FILTER CONCAT_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_concat STATIC ${CONCAT_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_concat PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_concat PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_concat PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE CONCAT_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_concat_test SRCS ${CONCAT_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_concat)