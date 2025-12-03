# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE layers_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/cpu/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*.cpp)
list(FILTER layers_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER layers_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
if(NOT WITH_CUDA)
  list(FILTER layers_SRCS EXCLUDE REGEX ".*/flashinfer_.*\\.cpp")
endif()

message(STATUS "layers_SRCS: ${layers_SRCS}")

set(layers_nvidia_SRCS "")
set(layers_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*.cpp)
  list(APPEND layers_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

set(layers_ascend_SRCS "")
set(layers_ascend_LIBS "")

if(WITH_ACL)
  file(GLOB_RECURSE layers_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*.cpp)
  list(APPEND layers_ascend_LIBS ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common
    atb_attention)
endif()

set(layers_zixiao_SRCS "")
set(layers_zixiao_LIBS "")

if(WITH_TOPS)
  file(GLOB_RECURSE layers_zixiao_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/zixiao/*.cpp)
  list(APPEND layers_zixiao_LIBS ${TOPS_SHARED_LIBS})
endif()

add_library(layers STATIC ${layers_SRCS} ${layers_nvidia_SRCS} ${layers_ascend_SRCS} ${layers_zixiao_SRCS})
target_link_libraries(layers PUBLIC 
  kernels ${layers_nvidia_LIBS} ${layers_ascend_LIBS} layer_progress_tracker data_hub
)

# for test
set(layers_test_nvidia_SRCS "")
set(layers_test_ascend_SRCS "")
set(layers_test_zixiao_SRCS "")

if(WITH_CUDA)
  file(GLOB_RECURSE layers_test_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/nvidia/*test.cpp)
endif()

if(WITH_ACL)
  file(GLOB_RECURSE layers_test_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/ascend/*test.cpp)
endif()

file(GLOB_RECURSE layers_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/layers/*test.cpp
  ${layers_test_nvidia_SRCS} ${layers_test_ascend_SRCS})
message(STATUS "layers_test_SRCS: ${layers_test_SRCS}")
cpp_test(layers_test SRCS ${layers_test_SRCS} DEPS layers runtime utils)
