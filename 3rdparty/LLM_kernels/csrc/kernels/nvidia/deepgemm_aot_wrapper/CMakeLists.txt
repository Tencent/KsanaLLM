# Copyright 2024 Tencent Inc.  All rights reserved.

# set masked multihead attention kernels target
file(GLOB_RECURSE DEEPGEMM_AOT_WRAPPER_SRCS *.cc)
list(FILTER DEEPGEMM_AOT_WRAPPER_SRCS EXCLUDE REGEX ".*test.cc")
add_library(llm_kernels_nvidia_kernel_deepgemm_aot_wrapper STATIC ${DEEPGEMM_AOT_WRAPPER_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_deepgemm_aot_wrapper PUBLIC llm_kernels_nvidia_kernel_gemm_wrapper ksana_llm_yaml fmt -lcublas -lcudart -lcublasLt)
set_property(TARGET llm_kernels_nvidia_kernel_deepgemm_aot_wrapper PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_deepgemm_aot_wrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
if(SM VERSION_GREATER_EQUAL "90")
  file(GLOB_RECURSE DEEPGEMM_AOT_WRAPPER_TEST_SRCS *test.cc)
  message(STATUS "DEEPGEMM_AOT_WRAPPER_TEST_SRCS ${DEEPGEMM_AOT_WRAPPER_TEST_SRCS}")
  cc_test(llm_kernels_nvidia_kernel_deepgemm_aot_wrapper_test SRCS ${DEEPGEMM_AOT_WRAPPER_TEST_SRCS} DEPS llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_deepgemm_aot_wrapper LINKS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${PYTHON_LIBRARY})
endif()