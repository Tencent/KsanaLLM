# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE GPTQ_MARLIN_SRCS *.cu)
list(FILTER GPTQ_MARLIN_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_gptq_marlin STATIC ${GPTQ_MARLIN_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_gptq_marlin PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_gptq_marlin PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_gptq_marlin PUBLIC fmt -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE ADD_TEST_SRCS marlin_wrapper_test.cu)
cc_test(llm_kernels_nvidia_kernel_marlin_test SRCS ${ADD_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_permute
    llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_gemm_wrapper
    llm_kernels_nvidia_kernel_machete
    llm_kernels_nvidia_kernel_gptq_marlin)