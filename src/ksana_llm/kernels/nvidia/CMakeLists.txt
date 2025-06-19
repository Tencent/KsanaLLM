
file(GLOB_RECURSE TRITON_SRCS *.cpp)
list(FILTER TRITON_SRCS EXCLUDE REGEX ".*test.cpp")

find_package(CUDA REQUIRED)

add_library(llm_kernels_nvidia_kernel_triton STATIC ${TRITON_SRCS})
target_link_libraries(llm_kernels_nvidia_kernel_triton PUBLIC utils ${TORCH_LIBRARIES} -lcuda -lcudart ksana_llm_libs
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_moe
    ${CUDA_LIBRARIES})

file(GLOB_RECURSE KERNEL_WRAPPER_TEST_SRCS kernel_wrapper_test.cpp)
cc_test(kernel_wrapper_test SRCS ${KERNEL_WRAPPER_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

file(GLOB_RECURSE GROUPED_TOPK_TEST_SRCS grouped_topk_test.cpp)
cc_test(grouped_topk_test SRCS ${GROUPED_TOPK_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton LINKS ${TORCH_LIBRARIES})

file(GLOB_RECURSE PER_TOKEN_GROUP_QUANT_FP8_TEST_SRCS per_token_group_quant_fp8_test.cpp)
cc_test(per_token_group_quant_fp8_test SRCS ${PER_TOKEN_GROUP_QUANT_FP8_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

file(GLOB_RECURSE FUSED_MOE_TEST_SRCS fused_moe_test.cpp)
cc_test(fused_moe_test SRCS ${FUSED_MOE_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

file(GLOB_RECURSE FUSED_MOE_GPTQ_AWQ_TEST_SRCS fused_moe_gptq_awq_test.cpp)
cc_test(fused_moe_gptq_awq_test SRCS ${FUSED_MOE_GPTQ_AWQ_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)

if (SM STREQUAL "90" OR SM STREQUAL "90a")
    file(GLOB_RECURSE FUSED_MOE_GPTQ_INT4_FP8_TEST_SRCS fused_moe_gptq_int4_fp8_test.cpp)
    cc_test(fused_moe_gptq_int4_fp8_test SRCS ${FUSED_MOE_GPTQ_INT4_FP8_TEST_SRCS} DEPS
        llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_triton)
endif()