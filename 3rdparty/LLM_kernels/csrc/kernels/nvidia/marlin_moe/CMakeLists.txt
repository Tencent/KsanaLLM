# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE MARLIN_MOE_SRCS *.cu marlin_kernels/*.cu)
list(FILTER MARLIN_MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_marlin_moe STATIC ${MARLIN_MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_marlin_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_marlin_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_marlin_moe PUBLIC -lcublas -lcudart -lcublasLt)
