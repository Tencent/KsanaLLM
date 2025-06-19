# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(external/trpc-cpp)
include(external/trpc-robus)

file(GLOB_RECURSE trpc_endpoint_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(FILTER trpc_endpoint_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "trpc_endpoint_SRCS: ${trpc_endpoint_SRCS}")

add_library(trpc_endpoint SHARED ${trpc_endpoint_SRCS})

target_link_libraries(trpc_endpoint PUBLIC
    trpc
    loguru
    trpc_robust
)

# for test
file(GLOB_RECURSE trpc_endpoint_test_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*test.cpp)
message(STATUS "trpc_endpoint_test_SRCS: ${trpc_endpoint_test_SRCS}")
cpp_test(trpc_endpoint_test
  SRCS ${trpc_endpoint_test_SRCS}
  DEPS trpc_endpoint)
