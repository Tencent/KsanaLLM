# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE ALL_REDUCE_SRCS *.cu)
list(FILTER ALL_REDUCE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_all_reduce STATIC ${ALL_REDUCE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_all_reduce PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_all_reduce PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_all_reduce PUBLIC -lcuda -lcudart -lnccl -lnvidia-ml)


# for test
file(GLOB_RECURSE ALL_REDUCE_TEST_SRCS *test.cu)
message(STATUS "ALL_REDUCE_TEST_SRCS : ${ALL_REDUCE_TEST_SRCS}")
cc_test(llm_kernels_nvidia_kernel_custom_all_reduce_test
  SRCS ${ALL_REDUCE_TEST_SRCS}
  DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_all_reduce)