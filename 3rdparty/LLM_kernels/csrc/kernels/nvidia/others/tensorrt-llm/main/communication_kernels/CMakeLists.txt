# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE TRT_ALL_REDUCE_SRCS *.cu)
list(FILTER TRT_ALL_REDUCE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_trt_all_reduce STATIC ${TRT_ALL_REDUCE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_trt_all_reduce PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_trt_all_reduce PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_trt_all_reduce PUBLIC -lcuda -lcudart -lnccl -lnvidia-ml
  llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE TRT_ALL_REDUCE_TEST_SRCS *test.cu)
message(STATUS "TRT_ALL_REDUCE_TEST_SRCS : ${TRT_ALL_REDUCE_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_trt_all_reduce_test SRCS ${TRT_ALL_REDUCE_TEST_SRCS} DEPS
  llm_kernels_nvidia_kernel_trt_all_reduce
  llm_kernels_nvidia_kernel_fused_add_norm
)
