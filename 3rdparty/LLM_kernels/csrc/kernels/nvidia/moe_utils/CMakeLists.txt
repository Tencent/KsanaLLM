# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE MOE_OPS_SRCS *.cu)
list(FILTER MOE_OPS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe_utils STATIC ${MOE_OPS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_moe_utils PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE MOE_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_moe_utils_test SRCS ${MOE_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_moe_utils)
