# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE EXPAND_SRCS *.cu)
list(FILTER EXPAND_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_expand STATIC ${EXPAND_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_expand PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_expand PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_expand PUBLIC -lcublas -lcudart -lcublasLt
                      llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE EXPAND_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_expand_test SRCS ${EXPAND_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_expand
    torch c10)
