# Copyright 2025 Tencent Inc.  All rights reserved.

if(SM VERSION_GREATER_EQUAL "90")
  file(GLOB_RECURSE PER_TOKEN_GROUP_QUANT_ALL_FILES
    per_token_group_quant_creator.sh
    per_token_group_quant_fp8.py
  )

  set(PER_TOKEN_GROUP_QUANT_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/per_token_group_quant_md5_sum)
  set(PER_TOKEN_GROUP_QUANT_MD5_FILE ${PER_TOKEN_GROUP_QUANT_MD5_SUM_DIR}/.per_token_group_quant.md5sums)

  set(PER_TOKEN_GROUP_QUANT_NEED_REGENERATE TRUE)

  if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
    set(PER_TOKEN_GROUP_QUANT_NEED_REGENERATE FALSE)
    message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping per_token_group_quant kernel generation")
  endif()

  if(EXISTS ${PER_TOKEN_GROUP_QUANT_MD5_FILE})
    file(READ ${PER_TOKEN_GROUP_QUANT_MD5_FILE} PREV_MD5)

    calculate_files_md5("${PER_TOKEN_GROUP_QUANT_ALL_FILES}" CURRENT_MD5)

    if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
      set(PER_TOKEN_GROUP_QUANT_NEED_REGENERATE FALSE)
      message(STATUS "Files MD5 unchanged, skipping regeneration")
    else()
      message(STATUS "Files MD5 changed, regenerating")
      message(STATUS "Previous MD5: ${PREV_MD5}")
      message(STATUS "Current MD5: ${CURRENT_MD5}")
    endif()
  endif()

  if(PER_TOKEN_GROUP_QUANT_NEED_REGENERATE)
    message(STATUS "Start generate per_token_group_quant triton kernel")

    execute_process(COMMAND bash per_token_group_quant_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE PER_TOKEN_GROUP_QUANT_RESULT_VARIABLE OUTPUT_VARIABLE PER_TOKEN_GROUP_QUANT_RESULT_OUTPUT)

    if(PER_TOKEN_GROUP_QUANT_RESULT_VARIABLE AND NOT PER_TOKEN_GROUP_QUANT_RESULT_VARIABLE EQUAL 0)
      message(FATAL_ERROR "bash per_token_group_quant_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${PER_TOKEN_GROUP_QUANT_RESULT_OUTPUT}")
    endif()

    calculate_files_md5("${PER_TOKEN_GROUP_QUANT_ALL_FILES}" NEW_MD5)
    file(WRITE ${PER_TOKEN_GROUP_QUANT_MD5_FILE} ${NEW_MD5})

    message(STATUS "Success generate per_token_group_quant triton kernel")
  endif()
endif()
