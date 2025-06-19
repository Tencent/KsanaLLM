# Copyright 2025 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE DECODE_GROUPED_ATTN_M_FWD_ALL_FILES
  decode_grouped_att_m_fwd_creator.sh
  decode_grouped_att_m_fwd.py
)

set(DECODE_GROUPED_ATTN_M_FWD_MD5_SUM_DIR ${CMAKE_CURRENT_BINARY_DIR}/decode_grouped_att_m_fwd_md5_sum)
set(DECODE_GROUPED_ATTN_M_FWD_MD5_FILE ${DECODE_GROUPED_ATTN_M_FWD_MD5_SUM_DIR}/.decode_grouped_att_m_fwd.md5sums)

set(DECODE_GROUPED_ATTN_M_FWD_NEED_REGENERATE TRUE)

if(DEFINED ENV{DISABLE_GENERATE_TRITON_KERNEL})
  set(DECODE_GROUPED_ATTN_M_FWD_NEED_REGENERATE FALSE)
  message(STATUS "DISABLE_GENERATE_TRITON_KERNEL is set, skipping decode_grouped_att_m_fwd kernel generation")
endif()

if(EXISTS ${DECODE_GROUPED_ATTN_M_FWD_MD5_FILE})
  file(READ ${DECODE_GROUPED_ATTN_M_FWD_MD5_FILE} PREV_MD5)

  calculate_files_md5("${DECODE_GROUPED_ATTN_M_FWD_ALL_FILES}" CURRENT_MD5)

  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(DECODE_GROUPED_ATTN_M_FWD_NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(DECODE_GROUPED_ATTN_M_FWD_NEED_REGENERATE)
  message(STATUS "Start generate decode_grouped_att_m_fwd triton kernel")

  execute_process(COMMAND bash decode_grouped_att_m_fwd_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE DECODE_GROUPED_ATTN_M_FWD_RESULT_VARIABLE OUTPUT_VARIABLE DECODE_GROUPED_ATTN_M_FWD_RESULT_OUTPUT)

  if(DECODE_GROUPED_ATTN_M_FWD_RESULT_VARIABLE AND NOT DECODE_GROUPED_ATTN_M_FWD_RESULT_VARIABLE EQUAL 0)
    message(FATAL_ERROR "bash decode_grouped_att_m_fwd_creator.sh ${CMAKE_CURRENT_SOURCE_DIR} ${TRITON_KERNEL_PATH} with error:\n${DECODE_GROUPED_ATTN_M_FWD_RESULT_OUTPUT}")
  endif()

  calculate_files_md5("${DECODE_GROUPED_ATTN_M_FWD_ALL_FILES}" NEW_MD5)
  file(WRITE ${DECODE_GROUPED_ATTN_M_FWD_MD5_FILE} ${NEW_MD5})

  message(STATUS "Success generate decode_grouped_att_m_fwd triton kernel")
endif()
