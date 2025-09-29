# Copyright 2024 Tencent Inc.  All rights reserved.

# set Nvidia utils target
file(GLOB_RECURSE NVIDIA_UTILS_SRCS *.cpp *.cu)
list(APPEND NVIDIA_UTILS_SRCS ../common.cc)
list(FILTER NVIDIA_UTILS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_utils STATIC ${NVIDIA_UTILS_SRCS})
target_compile_definitions(llm_kernels_nvidia_utils PRIVATE FLASHINFER_USE_FP8=1)
target_link_libraries(llm_kernels_nvidia_utils PUBLIC -lcublas -lcudart -lcurand -lnvidia-ml)
set_property(TARGET llm_kernels_nvidia_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# for test
file(GLOB_RECURSE NVIDIA_UTILS_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_utils_test SRCS ${NVIDIA_UTILS_TEST_SRCS} DEPS
     llm_kernels_nvidia_utils)
