# Copyright 2025 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE MOE_WNA16_SRCS *.cu)
list(FILTER MOE_WNA16_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe_wna16 STATIC ${MOE_WNA16_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe_wna16 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe_wna16 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_moe_wna16 PUBLIC -lcublas -lcudart -lcublasLt)