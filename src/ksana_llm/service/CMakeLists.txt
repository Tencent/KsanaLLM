# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE service_SRCS
	${PROJECT_SOURCE_DIR}/src/ksana_llm/service/*.cpp)
list(FILTER service_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "service_SRCS: ${service_SRCS}")

add_library(service STATIC ${service_SRCS})

# for test
file(GLOB_RECURSE service_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/service/*test.cpp)

message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(service_test SRCS ${service_test_SRCS} DEPS service)