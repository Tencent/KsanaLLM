# Copyright 2024 Tencent Inc.  All rights reserved.

# set masked multihead attention kernels target
file(GLOB_RECURSE PAGED_ATTENTION_SRCS *.cu)
list(FILTER PAGED_ATTENTION_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_paged_attention STATIC ${PAGED_ATTENTION_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_paged_attention PUBLIC -lcudart)
set_property(TARGET llm_kernels_nvidia_kernel_paged_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_paged_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_definitions(llm_kernels_nvidia_kernel_paged_attention PRIVATE FLASHINFER_USE_FP8=1)

# for test
file(GLOB_RECURSE PAGED_ATTENTION_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_paged_attention_test SRCS ${PAGED_ATTENTION_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils    
    llm_kernels_nvidia_kernel_paged_attention)
