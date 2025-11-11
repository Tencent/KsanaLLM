# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE WEIGHT_SCALE_SRCS *.cu)
list(FILTER WEIGHT_SCALE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_weight_scale STATIC ${WEIGHT_SCALE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_weight_scale PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_weight_scale PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_weight_scale PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE WEIGHT_SCALE_TEST_SRCS *test.cu)
if(WEIGHT_SCALE_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_weight_scale_test SRCS ${WEIGHT_SCALE_TEST_SRCS} DEPS 
      llm_kernels_nvidia_utils
      llm_kernels_nvidia_kernel_weight_scale)
endif()