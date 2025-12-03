# Copyright 2025 Tencent Inc.  All rights reserved.

# set flashinfer prefill attention kernels target
file(GLOB_RECURSE FLASHINFER_ATTENTION_SRCS *.cu)
list(FILTER FLASHINFER_ATTENTION_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_flashinfer_attention STATIC ${FLASHINFER_ATTENTION_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_flashinfer_attention PRIVATE prefill_kernels)
set_property(TARGET llm_kernels_nvidia_kernel_flashinfer_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_flashinfer_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
file(GLOB_RECURSE FLASHINFER_ATTENTION_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_flashinfer_attention_test SRCS ${FLASHINFER_ATTENTION_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_flashinfer_attention)
