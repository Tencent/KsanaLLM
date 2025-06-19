# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE GROUPED_TOPK_SRCS *.cu)
list(FILTER GROUPED_TOPK_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_grouped_topk STATIC ${GROUPED_TOPK_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_grouped_topk PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_grouped_topk PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_grouped_topk PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE GROUPED_TOPK_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_grouped_topk_test SRCS ${GROUPED_TOPK_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_grouped_topk)