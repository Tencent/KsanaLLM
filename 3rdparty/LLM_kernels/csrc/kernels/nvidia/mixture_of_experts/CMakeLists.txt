# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE MIXTURE_OF_EXPERTS_SRCS *.cu)
list(FILTER MIXTURE_OF_EXPERTS_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_mixture_of_experts STATIC ${MIXTURE_OF_EXPERTS_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_mixture_of_experts PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_mixture_of_experts PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_mixture_of_experts PUBLIC -lcublas -lcudart -lcublasLt
    moe_gemm_src llm_kernels_nvidia_kernel_asymmetric_gemm)

# for test
file(GLOB_RECURSE MIXTURE_OF_EXPERTS_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_mixture_of_experts_test SRCS ${MIXTURE_OF_EXPERTS_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_mixture_of_experts)