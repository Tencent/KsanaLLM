# Copyright 2025 Tencent Inc.  All rights reserved.

if (SM STREQUAL "90" OR SM STREQUAL "90a")
  add_definitions(-DENABLE_COMMON_INT4_FP8_DEQUANT=ON)

  file(GLOB_RECURSE FUSED_MOE_GPTQ_INT4_FP8_ALL_FILES
    fused_moe_gptq_int4_fp8_kernel_creator.sh
    fused_moe_gptq_int4_fp8_kernel.py
  )

  set(FUSED_MOE_GPTQ_INT4_FP8_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/fused_moe_gptq_int4_fp8_kernel_md5_sum)
  set(FUSED_MOE_GPTQ_INT4_FP8_MD5_FILE ${FUSED_MOE_GPTQ_INT4_FP8_MD5_SUM_DIR}/.fused_moe_gptq_int4_fp8_kernel.md5sums)

  set(FUSED_MOE_GPTQ_INT4_FP8_NEED_REGENERATE TRUE)

  if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
    set(FUSED_MOE_GPTQ_INT4_FP8_NEED_REGENERATE FALSE)
    message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping fused_moe_gptq_awq kernel generation")
  endif()

  if(EXISTS ${FUSED_MOE_GPTQ_INT4_FP8_MD5_FILE})
    file(READ ${FUSED_MOE_GPTQ_INT4_FP8_MD5_FILE} PREV_MD5)

    calculate_files_md5("${FUSED_MOE_GPTQ_INT4_FP8_ALL_FILES}" CURRENT_MD5)

    if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
      set(FUSED_MOE_GPTQ_INT4_FP8_NEED_REGENERATE FALSE)
      message(STATUS "Files MD5 unchanged, skipping regeneration")
    else()
      message(STATUS "Files MD5 changed, regenerating")
      message(STATUS "Previous MD5: ${PREV_MD5}")
      message(STATUS "Current MD5: ${CURRENT_MD5}")
    endif()
  endif()

  if(FUSED_MOE_GPTQ_INT4_FP8_NEED_REGENERATE)
    message(STATUS "Start generate fused_moe_gptq_int4_fp8_kernel triton kernel")

    execute_process(COMMAND bash fused_moe_gptq_int4_fp8_kernel_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE FUSED_MOE_GPTQ_INT4_FP8_RESULT_VARIABLE OUTPUT_VARIABLE FUSED_MOE_GPTQ_INT4_FP8_RESULT_OUTPUT)

    if(FUSED_MOE_GPTQ_INT4_FP8_RESULT_VARIABLE AND NOT FUSED_MOE_GPTQ_INT4_FP8_RESULT_VARIABLE EQUAL 0)
      message(FATAL_ERROR "bash fused_moe_gptq_int4_fp8_kernel_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${FUSED_MOE_GPTQ_INT4_FP8_RESULT_OUTPUT}")
    endif()

    calculate_files_md5("${FUSED_MOE_GPTQ_INT4_FP8_ALL_FILES}" NEW_MD5)
    file(WRITE ${FUSED_MOE_GPTQ_INT4_FP8_MD5_FILE} ${NEW_MD5})
    
    message(STATUS "Success generate fused_moe_gptq_int4_fp8_kernel triton kernel")
  endif()

endif()

# set kernels target
file(GLOB_RECURSE DEQUANT_SRCS *.cu)
list(FILTER DEQUANT_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_dequant STATIC ${DEQUANT_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_dequant PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_dequant PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_dequant PUBLIC -lcublas -lcudart -lcublasLt llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE DEQUANT_TEST_SRCS dequant_test.cu)
cc_test(llm_kernels_nvidia_kernel_dequant_test SRCS ${DEQUANT_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_dequant)