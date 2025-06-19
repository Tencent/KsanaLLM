# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ASSEMBLE_TOKENS_HIDDEN_SRCS *.cu)
list(FILTER ASSEMBLE_TOKENS_HIDDEN_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_assemble_tokens_hidden STATIC ${ASSEMBLE_TOKENS_HIDDEN_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_assemble_tokens_hidden PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_assemble_tokens_hidden PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_assemble_tokens_hidden PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE ASSEMBLE_TOKENS_HIDDEN_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_assemble_tokens_hidden_test SRCS ${ASSEMBLE_TOKENS_HIDDEN_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_assemble_tokens_hidden)