# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

include(external/opentelemetry)
file(GLOB_RECURSE profiler_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*.cpp)

list(FILTER profiler_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "profiler_SRCS: ${profiler_SRCS}")


add_library(profiler STATIC ${profiler_SRCS})
# 添加 OpenTelemetry 的包含目录
target_include_directories(profiler
  PUBLIC
    ${OPENTELEMETRY_CPP_INCLUDE_DIRS}  # 添加 OpenTelemetry 的包含目录
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "OPENTELEMETRY_CPP_INCLUDE_DIRS: ${OPENTELEMETRY_CPP_INCLUDE_DIRS}")
target_link_libraries(profiler
    PUBLIC
      ${OPENTELEMETRY_CPP_LIBRARIES}
    PRIVATE
      TBB::tbb
)

# for test
file(GLOB_RECURSE profiler_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/profiler/*test.cpp)
cpp_test(profiler_test SRCS ${profiler_test_SRCS} DEPS profiler)
