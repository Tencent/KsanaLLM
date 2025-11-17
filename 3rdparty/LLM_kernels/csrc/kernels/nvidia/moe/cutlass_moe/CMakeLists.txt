# Copyright 2025 Tencent Inc.  All rights reserved.

if (SM STREQUAL "90" OR SM STREQUAL "90a")
    set(ENABLE_CUTLASSMOE 1)
else()
    set(ENABLE_CUTLASSMOE 0)
    message(WARNING "SM != 90, set ENABLE_CUTLASSMOE=0")
endif()

# set kernels target
file(GLOB_RECURSE CUTLASS_MOE_SRCS *.cu *.cpp)
list(FILTER CUTLASS_MOE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_moe_cutlass_moe STATIC ${CUTLASS_MOE_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_moe_cutlass_moe PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_moe_cutlass_moe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC ENABLE_CUTLASSMOE=${ENABLE_CUTLASSMOE})
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE NDEBUG)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE ENABLE_BF16)
target_compile_definitions(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE ENABLE_FP4)
target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC -lcublas -lcudart -lcublasLt
                      llm_kernels_nvidia_utils)
if (ENABLE_CUTLASSMOE)
    target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PUBLIC llm_kernels_nvidia_kernel_others_trtllm_dev_thop)
endif()
target_link_libraries(llm_kernels_nvidia_kernel_moe_cutlass_moe PRIVATE cutlass_c4x)