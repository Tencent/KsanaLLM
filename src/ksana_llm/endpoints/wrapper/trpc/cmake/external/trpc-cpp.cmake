# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(TRPC_CPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/trpc-cpp)

FetchContent_Declare(trpc_cpp
  GIT_REPOSITORY https://git.woa.com/trpc-cpp/trpc-cpp.git
  GIT_TAG v0.19.5
  SOURCE_DIR ${TRPC_CPP_INSTALL_DIR}
)

# Add library alias for trpc-cpp
if(TARGET fmt)
  add_library(trpc_fmt ALIAS fmt)
endif()

if(TARGET yaml-cpp)
  add_library(trpc_yaml ALIAS yaml-cpp)
endif()

if(TARGET gflags)
  add_library(trpc_gflags ALIAS gflags_nothreads_static)
endif()

if(TARGET gtest)
  add_library(trpc_gmock ALIAS gmock)
  add_library(trpc_gtest ALIAS gtest)
endif()

if(TARGET protoc)
  add_library(trpc-protobuf ALIAS libprotobuf)
  add_library(trpc-protoc ALIAS libprotoc)
  add_library(trpc_protobuf ALIAS libprotobuf)
  add_library(trpc_protoc ALIAS libprotoc)
endif()

FetchContent_GetProperties(trpc_cpp)

if(NOT trpc_cpp_POPULATED)
  FetchContent_Populate(trpc_cpp)

  set(TRPC_BUILD_WITH_CPP_20 OFF)
  set(TRPC_BUILD_WITH_NAME_POLARIS ON)
  set(TRPC_BUILD_WITH_METRICS_007 ON)
  execute_process(COMMAND
    sh -c "git diff --quiet && git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/external/trpc-cpp.patch"
    WORKING_DIRECTORY ${trpc_cpp_SOURCE_DIR})
  execute_process(COMMAND
    sh -c "grep -rl 'add_definitions(-std=c++17)' ./ | xargs sed -i '/add_definitions(-std=c++17)/d'"
    WORKING_DIRECTORY ${trpc_cpp_SOURCE_DIR}/cmake)

  # NOTE(karlluo): trpc v0.19.5 has error in C++20, for close it under NPU environment.
  execute_process(COMMAND
    sh -c "grep -rl 'set(TRPC_BUILD_WITH_CPP20 ON)' ./ | xargs sed -i '/set(TRPC_BUILD_WITH_CPP20 ON)/d'"
    WORKING_DIRECTORY ${trpc_cpp_SOURCE_DIR})

  add_subdirectory(${trpc_cpp_SOURCE_DIR} ${trpc_cpp_BINARY_DIR})
endif()

message(STATUS "Trpc-cpp source directory: ${trpc_cpp_SOURCE_DIR}")
message(STATUS "Trpc-cpp binary directory: ${trpc_cpp_BINARY_DIR}")

include_directories(${trpc_cpp_SOURCE_DIR})
