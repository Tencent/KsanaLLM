# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE UTILS_SRCS *.cu *.cpp)
list(FILTER UTILS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_others_trtllm_dev_utils STATIC ${UTILS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_others_trtllm_dev_utils PUBLIC -lcublas -lcudart -lcublasLt)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_utils PRIVATE NDEBUG)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_utils PRIVATE ENABLE_BF16)