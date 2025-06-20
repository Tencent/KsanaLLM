# Copyright 2024 Tencent Inc.  All rights reserved.

if(SM VERSION_GREATER_EQUAL "90")
    add_definitions(-DENABLE_FLASHINFER=ON)
endif()

# set kernels target
file(GLOB_RECURSE MOE_SRCS *.cu)
list(FILTER MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe STATIC ${MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_moe PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE MOE_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_moe_test SRCS ${MOE_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_moe)
