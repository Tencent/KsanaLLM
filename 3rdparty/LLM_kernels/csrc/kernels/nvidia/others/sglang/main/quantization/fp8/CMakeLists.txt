# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE QUANT_FP8_SRCS *.cu)
list(FILTER QUANT_FP8_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_quant_fp8 STATIC ${QUANT_FP8_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_quant_fp8 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_quant_fp8 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_quant_fp8 PUBLIC -lcublas -lcudart -lcublasLt
    llm_kernels_nvidia_utils)

file(GLOB_RECURSE QUANT_FP8_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_quant_fp8_test SRCS ${QUANT_FP8_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_activation
    llm_kernels_nvidia_kernel_quant_fp8)
