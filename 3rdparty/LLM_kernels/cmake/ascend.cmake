# Copyright 2024 Tencent Inc.  All rights reserved.
if(NOT WITH_ACL)
  return()
endif()

find_program(AWK awk mawk gawk)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")

# ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT "$ENV{ASCEND_MAIN_PATH}" STREQUAL "")
  set(ASCEND_MAIN_PATH $ENV{ASCEND_MAIN_PATH})
else()
  set(ASCEND_MAIN_PATH "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux")
endif()

if(NOT "$ENV{ATB_HOME_PATH}" STREQUAL "")
  set(ATB_HOME_PATH $ENV{ATB_HOME_PATH})
else()
  set(ATB_HOME_PATH "/usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/atb")
endif()

set(CANN_VER "")

# get cann version
if(EXISTS "${ASCEND_PATH}/version.cfg")
  # awk -F= '{ gsub(/\[/, "", $0); gsub(/\]/, "", $0); if ($1 == "runtime_running_version") print $2 }' /usr/local/Ascend/ascend-toolkit/latest/version.cfg
  execute_process(COMMAND /usr/bin/bash -c "\"${AWK}\" -F= '\{ gsub(/\\[/, \"\", $0); gsub(/\\]/, \"\", $0); if ($1 == \"runtime_running_version\") print $2 \}' ${ASCEND_PATH}/version.cfg" OUTPUT_VARIABLE CANN_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using CANN ${CANN_VER} at ${ASCEND_PATH}")

  if("${CANN_VER}" STREQUAL "")
    message(FATAL_ERROR "Can not find CANN version file in ${ASCEND_PATH}/version.cfg, please check CANN install and config")
  endif()
else()
  message(FATAL_ERROR "Can not find CANN version file in ${ASCEND_PATH}/version.cfg, please check CANN install and config")
endif()

# get ATB version
set(ATB_VER "")

# for CANN8 POC530 ATB RC1
if(EXISTS "${ATB_HOME_PATH}/set_env.sh")
  add_definitions("-DENABLE_ACL_ATB")

  # awk -F: '{ gsub(/ /, "", $0); if ($1 == "Ascend-mindie-atbVersion") print $2 }' /usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/version.info
  execute_process(COMMAND /bin/sh -c "\"${AWK}\" -F: '\{ gsub(/ /, \"\", $0); if ($1 == \"Ascend-mindie-atbVersion\") print $2 \}' ${ATB_HOME_PATH}/../version.info" OUTPUT_VARIABLE ATB_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using ATB ${ATB_VER} at ${ATB_HOME_PATH}")

# for CANN8 RC2 ATB RC2
elseif(EXISTS "${ATB_HOME_PATH}/../../version.info")
  add_definitions("-DENABLE_ACL_ATB")

  # awk -F: '{ gsub(/ /, "", $0); if ($1 == "Ascend-mindie-atbVersion") print $2 }' /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/../../version.info
  execute_process(COMMAND /bin/sh -c "\"${AWK}\" -F: '\{ gsub(/ /, \"\", $0); if ($1 == \"Ascend-cann-atb\") print $2 \}' ${ATB_HOME_PATH}/../../version.info" OUTPUT_VARIABLE ATB_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using ATB ${ATB_VER} at ${ATB_HOME_PATH}")
endif()

set(CCE_CMAKE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

set(ACL_INC_DIRS
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
  ${ASCEND_MAIN_PATH}/tikcpp/tikcfw
  ${ATB_HOME_PATH}/include
)

set(ACL_LIB_DIRS
  ${ASCEND_PATH}/lib64
  ${ATB_HOME_PATH}/lib
)

set(ACL_SHARED_LIBS
  ${ASCEND_PATH}/lib64/libascendcl.so
  ${ASCEND_PATH}/lib64/libnnopbase.so
  ${ASCEND_PATH}/lib64/libopapi.so
  ${ASCEND_PATH}/lib64/libascendc_runtime.a
  ${ASCEND_PATH}/lib64/libruntime.so
  ${ASCEND_PATH}/lib64/libtiling_api.a
  ${ASCEND_PATH}/x86_64-linux/lib64/libregister.so
  ${ASCEND_PATH}/x86_64-linux/lib64/libplatform.so
  ${ASCEND_PATH}/lib64/libhccl.so
)

# for CANN8 POC530 ATB RC1
if(EXISTS "${ATB_HOME_PATH}/set_env.sh")
  list(APPEND ACL_SHARED_LIBS
    ${ATB_HOME_PATH}/lib/libatb.so
    ${ATB_HOME_PATH}/lib/libasdops.so
    ${ATB_HOME_PATH}/lib/liblcal.so)
endif()

# for CANN8 RC2 ATB RC2
if(EXISTS "${ATB_HOME_PATH}/../../version.info")
  list(APPEND ACL_SHARED_LIBS
    ${ATB_HOME_PATH}/lib/libatb.so
    ${ATB_HOME_PATH}/lib/libasdops.so
    ${ATB_HOME_PATH}/lib/liblcal.so)
endif()

# awk -F= '{ if ($1 == "version_dir") print $2 }' /usr/local/Ascend/ascend-toolkit/latest/toolkit/version.info | awk -F. '{ print $1 }'
execute_process(COMMAND /bin/sh -c "\"${AWK}\" -F= '\{ if ($1 == \"version_dir\") print $2 \}' ${ASCEND_PATH}/toolkit/version.info | awk -F. '\{ print $1 \}'" OUTPUT_VARIABLE ASCEND_TOOLKIT_MAR_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions("-DASCEND_TOOLKIT_MAR_VER_${ASCEND_TOOLKIT_MAR_VER}")
