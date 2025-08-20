# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ADD_MUL_SRCS *.cu)
list(FILTER ADD_MUL_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_add_mul STATIC ${ADD_MUL_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_add_mul PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_add_mul PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_add_mul PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE ADD_MUL_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_add_mul_test SRCS ${ADD_MUL_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_add_mul) 