# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE WEIGHT_ONLY_BATCHED_GEMV_SRCS *.cu)
list(FILTER WEIGHT_ONLY_BATCHED_GEMV_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_weight_only_batched_gemv STATIC ${WEIGHT_ONLY_BATCHED_GEMV_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_weight_only_batched_gemv PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_weight_only_batched_gemv PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_weight_only_batched_gemv PUBLIC -lcublas -lcudart -lcublasLt)
target_link_libraries(llm_kernels_nvidia_kernel_weight_only_batched_gemv PRIVATE cutlass_c3x)

# for test
# file(GLOB_RECURSE WEIGHT_ONLY_BATCHED_GEMV_TEST_SRCS *test.cu)
# cc_test(llm_kernels_nvidia_kernel_weight_only_batched_gemv_test SRCS ${WEIGHT_ONLY_BATCHED_GEMV_TEST_SRCS} DEPS
#     llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_weight_only_batched_gemv)