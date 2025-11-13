# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE THOP_SRCS *.cu *.cpp)
list(FILTER THOP_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_others_trtllm_dev_thop STATIC ${THOP_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_thop PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_thop PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_others_trtllm_dev_thop PUBLIC fmt -lcublas -lcudart -lcublasLt
                      cutlass_c4x
                      llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_others_trtllm_dev_common
                      llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels)

# for test
cc_test(llm_kernels_nvidia_kernel_others_trtllm_dev_thop_moeOp_test SRCS moeOp_test.cu DEPS 
    llm_kernels_nvidia_kernel_others_trtllm_dev_thop
    torch c10)
cc_test(llm_kernels_nvidia_kernel_others_trtllm_dev_thop_finegrained_mixed_dtype_gemm_thop_test SRCS finegrained_mixed_dtype_gemm_thop_test.cu DEPS 
    llm_kernels_nvidia_kernel_others_trtllm_dev_thop
    torch c10)