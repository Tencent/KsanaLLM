# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE samplers_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/base/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/beam_search/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/topk/*.cpp)
list(FILTER samplers_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "samplers_SRCS: ${samplers_SRCS}")

set(samplers_nvidia_LIBS "")
set(samplers_ascend_LIBS "")

if(WITH_CUDA)
  list(APPEND samplers_nvidia_LIBS cudart llm_kernels_nvidia_kernel_samplers)
endif()

if(WITH_ACL)
  list(APPEND samplers_ascend_LIBS ${ACL_SHARED_LIBS} ${ACL_SHARED_LIBS})
endif()

add_library(samplers STATIC ${samplers_SRCS})
target_link_libraries(samplers PUBLIC cache_manager utils ${samplers_nvidia_LIBS} ${samplers_ascend_LIBS})

# for test
file(GLOB_RECURSE samplers_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/samplers/*test.cpp)
cpp_test(samplers_test SRCS ${samplers_test_SRCS} DEPS samplers)
