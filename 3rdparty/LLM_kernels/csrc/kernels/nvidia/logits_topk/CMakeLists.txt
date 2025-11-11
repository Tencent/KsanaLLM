# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE LOGITS_TOPK_SRCS *.cu)
list(FILTER LOGITS_TOPK_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_logits_topk STATIC ${LOGITS_TOPK_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_logits_topk PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_logits_topk PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_logits_topk PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE LOGITS_TOPK_TEST_SRCS *test.cu)
if(LOGITS_TOPK_TEST_SRCS)
  cc_test(llm_kernels_nvidia_kernel_logits_topk_test SRCS ${LOGITS_TOPK_TEST_SRCS} DEPS 
      llm_kernels_nvidia_utils
      llm_kernels_nvidia_kernel_logits_topk)
endif()