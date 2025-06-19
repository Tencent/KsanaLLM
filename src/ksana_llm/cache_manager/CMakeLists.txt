# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE cache_manager_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/cache_manager/block_allocator/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/cache_manager/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/cache_manager/*.cpp)

list(FILTER cache_manager_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "cache_manager_SRCS: ${cache_manager_SRCS}")

add_library(cache_manager STATIC ${cache_manager_SRCS})
target_link_libraries(cache_manager PUBLIC utils)

# cache_manager_test
file(GLOB_RECURSE cache_manager_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/cache_manager/*test.cpp)

message(STATUS "cache_manager_test_SRCS: ${cache_manager_test_SRCS}")
cpp_test(cache_manager_test SRCS ${cache_manager_test_SRCS} DEPS cache_manager runtime)

# cache_block_allocator_test
file(GLOB_RECURSE cache_block_allocator_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/cache_manager/block_allocator/*test.cpp)

message(STATUS "cache_block_allocator_test_SRCS: ${cache_block_allocator_test_SRCS}")
cpp_test(cache_block_allocator_test SRCS ${cache_block_allocator_test_SRCS} DEPS cache_manager runtime)
