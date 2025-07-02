# Copyright 2024 Tencent Inc.All rights reserved.

# set kernels target
file(GLOB_RECURSE FUSED_ADD_NORM_SRCS *.cu)
list(FILTER FUSED_ADD_NORM_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_fused_add_norm STATIC ${FUSED_ADD_NORM_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_fused_add_norm PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_fused_add_norm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_fused_add_norm PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE FUSED_ADD_NORM_TEST_SRCS *test.cu)
# # copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/fused_add_norm/fused_add_norm_test.py ${CMAKE_CURRENT_BINARY_DIR})
# # copy python script for standalone test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/fused_add_norm/fused_add_norm_test.py ${CMAKE_BINARY_DIR})
cc_test(llm_kernels_nvidia_kernel_fused_add_norm_test SRCS ${FUSED_ADD_NORM_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_fused_add_norm)