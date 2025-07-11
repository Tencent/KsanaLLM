# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE multi_batch_controller_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/multi_batch_controller/*.cpp)
list(FILTER multi_batch_controller_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "multi_batch_controller_SRCS: ${multi_batch_controller_SRCS}")

add_library(multi_batch_controller STATIC ${multi_batch_controller_SRCS})

# multi_batch_controller_test
file(GLOB_RECURSE multi_batch_controller_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/multi_batch_controller/multi_batch_controller_test.cpp)

message(STATUS "multi_batch_controller_test_SRCS: ${multi_batch_controller_test_SRCS}")
cpp_test(multi_batch_controller_test SRCS ${multi_batch_controller_test_SRCS} DEPS data_hub)
