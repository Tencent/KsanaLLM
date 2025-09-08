# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE EXPERT_MOE_SRCS *.cu)
list(FILTER EXPERT_MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe_expert_map STATIC ${EXPERT_MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe_expert_map PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe_expert_map PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_moe_expert_map PUBLIC -lcublas -lcudart -lcublasLt
    llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE EXPERT_MAP_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_moe_expert_map_test SRCS ${EXPERT_MAP_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_moe_expert_map
    torch c10)