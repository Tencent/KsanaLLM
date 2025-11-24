# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE TOKENWEAVE_SRCS *.cu)
list(FILTER TOKENWEAVE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_tokenweave STATIC ${TOKENWEAVE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_tokenweave PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_tokenweave PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_tokenweave PUBLIC -lcuda ${NCCL_LIBRARIES} -lnvidia-ml
    llm_kernels_nvidia_utils torch c10)

file(GLOB_RECURSE TOKENWEAVE_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_tokenweave_test SRCS ${TOKENWEAVE_TEST_SRCS} DEPS
    llm_kernels_nvidia_kernel_tokenweave
    llm_kernels_nvidia_kernel_fused_add_norm)
