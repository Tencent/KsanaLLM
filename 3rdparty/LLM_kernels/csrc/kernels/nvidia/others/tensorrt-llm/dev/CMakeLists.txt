# Copyright 2025 Tencent Inc.  All rights reserved.

if(${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
    add_subdirectory(common)
    add_subdirectory(utils)
    add_subdirectory(cutlass_kernels)
    add_subdirectory(thop)

    if (SM STREQUAL "90" OR SM STREQUAL "90a")
        file(GLOB_RECURSE DEEP_GEMM_TEST_SRCS deepgemm_test.cu)
        cc_test(llm_kernels_nvidia_kernel_others_trtllm_dev_deepgemm_test SRCS ${DEEP_GEMM_TEST_SRCS} DEPS
            llm_kernels_nvidia_utils
            llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels
            torch c10
            cutlass_c4x)
    endif()
endif()