# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE IDENTITY_SRCS *.cu)
list(FILTER IDENTITY_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_identity STATIC ${IDENTITY_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_identity PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_identity PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_identity PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE IDENTITY_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_identity_test SRCS ${IDENTITY_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_identity)