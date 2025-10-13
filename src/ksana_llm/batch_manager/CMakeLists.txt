# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE batch_manager_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_manager/*.cpp)
list(FILTER batch_manager_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "batch_manager_SRCS: ${batch_manager_SRCS}")

add_library(batch_manager STATIC ${batch_manager_SRCS})

# batch_manager_test
cpp_test(batch_manager_test SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_manager/batch_manager_test.cpp
  DEPS runtime data_hub)

cpp_test(schedule_processor_test SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_manager/schedule_processor_test.cpp
  DEPS runtime data_hub)
