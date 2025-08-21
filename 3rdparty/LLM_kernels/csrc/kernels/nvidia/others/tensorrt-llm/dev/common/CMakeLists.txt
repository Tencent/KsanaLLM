# Copyright 2025 Tencent Inc.  All rights reserved.

add_definitions(-DENABLE_FP4=ON)

file(GLOB_RECURSE COMMON_SRCS *.cu *.cpp)
list(FILTER COMMON_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_others_trtllm_dev_common STATIC ${COMMON_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_common PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_common PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_others_trtllm_dev_common PUBLIC -lcublas -lcudart -lcublasLt)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_common PRIVATE NDEBUG)