# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE transfer_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/transfer/*.cpp)
list(FILTER transfer_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "transfer_SRCS: ${transfer_SRCS}")

add_library(transfer STATIC ${transfer_SRCS})
target_link_libraries(transfer PUBLIC utils connector)

set(MODELS_TEST_DEPS transfer runtime data_hub)

cpp_test(transfer_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/transfer/transfer_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
