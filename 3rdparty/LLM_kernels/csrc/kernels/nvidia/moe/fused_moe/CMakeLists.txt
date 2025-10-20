# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE FUSED_MOE_ALL_FILES
  fused_moe_creator.sh
  fused_moe.py
)

set(FUSED_MOE_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/fused_moe_md5_sum)
set(FUSED_MOE_MD5_FILE ${FUSED_MOE_MD5_SUM_DIR}/.fused_moe.md5sums)
set(FUSED_MOE_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(FUSED_MOE_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping fused_moe kernel generation")
endif()

if(EXISTS ${FUSED_MOE_MD5_FILE})
  file(READ ${FUSED_MOE_MD5_FILE} PREV_MD5)

  calculate_files_md5("${FUSED_MOE_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(FUSED_MOE_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(FUSED_MOE_NEED_REGENERATE)
  message(STATUS "Start generate fused_moe triton kernel")

  execute_process(COMMAND bash fused_moe_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE FUSED_MOE_RESULT_VARIABLE OUTPUT_VARIABLE FUSED_MOE_RESULT_OUTPUT)

  if(FUSED_MOE_RESULT_VARIABLE AND NOT FUSED_MOE_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "bash fused_moe_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${FUSED_MOE_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${FUSED_MOE_ALL_FILES}" NEW_MD5)
  file(WRITE ${FUSED_MOE_MD5_FILE} ${NEW_MD5})
  message(STATUS "Success generate fused_moe triton kernel")
endif()

# for test
file(GLOB_RECURSE fused_moe_test_SRCS
${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/*test.cpp)
# copy python script for make test
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/fused_moe.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/fused_moe/fused_moe.py ${CMAKE_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/common.py ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/csrc/kernels/nvidia/moe/common.py ${CMAKE_BINARY_DIR})
cc_test(fused_moe_config_find_test SRCS ${fused_moe_test_SRCS})

