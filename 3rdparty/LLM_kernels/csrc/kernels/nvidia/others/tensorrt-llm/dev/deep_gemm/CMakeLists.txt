add_definitions(-DENABLE_FP4=ON)

file(GLOB_RECURSE DEEP_GEMM_TEST_SRCS deepgemm_test.cu)
cc_test(llm_kernels_nvidia_kernel_others_trtllm_dev_deepgemm_test SRCS ${DEEP_GEMM_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels
    torch c10
    cutlass_c4x)