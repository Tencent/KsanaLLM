# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE data_hub_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/*.cpp)

list(FILTER data_hub_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER data_hub_SRCS EXCLUDE REGEX ".*/deepep_wrapper/.*")
message(STATUS "data_hub_SRCS: ${data_hub_SRCS}")

add_library(data_hub STATIC ${data_hub_SRCS})
target_link_libraries(data_hub PUBLIC utils runtime)

# schedule_output_test
file(GLOB_RECURSE schedule_output_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/schedule_output_test.cpp)

message(STATUS "schedule_output_test_SRCS: ${schedule_output_test_SRCS}")
cpp_test(data_hub_schedule_output_test SRCS ${schedule_output_test_SRCS} DEPS utils runtime data_hub)

# hidden_unit_buffer_test
file(GLOB_RECURSE hidden_unit_buffer_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/hidden_unit_buffer_test.cpp)

message(STATUS "hidden_unit_buffer_test_SRCS: ${hidden_unit_buffer_test_SRCS}")
cpp_test(data_hub_hidden_unit_buffer_test SRCS ${hidden_unit_buffer_test_SRCS} DEPS utils runtime)

# expert_parallel_hidden_unit_buffer_test
file(GLOB_RECURSE expert_parallel_hidden_unit_buffer_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/expert_parallel_hidden_unit_buffer_test.cpp)

message(STATUS "expert_parallel_hidden_unit_buffer_test_SRCS: ${expert_parallel_hidden_unit_buffer_test_SRCS}")
cpp_test(expert_parallel_hidden_unit_buffer_test SRCS ${expert_parallel_hidden_unit_buffer_test_SRCS} DEPS utils runtime)


# data_hub_test
file(GLOB_RECURSE data_hub_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/data_hub_test.cpp)

message(STATUS "data_hub_test_SRCS: ${data_hub_test_SRCS}")
cpp_test(data_hub_test SRCS ${data_hub_test_SRCS} DEPS data_hub utils runtime)

# expert_parallel_data_hub_test
file(GLOB_RECURSE expert_parallel_data_hub_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/expert_data_hub_test.cpp)

message(STATUS "expert_parallel_data_hub_test: ${expert_parallel_data_hub_test_SRCS}")
cpp_test(expert_parallel_data_hub_test SRCS ${expert_parallel_data_hub_test_SRCS} DEPS data_hub utils runtime)

# expert_parallel_deepep_wrapper_test
file(GLOB_RECURSE expert_parallel_deepep_wrapper_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/data_hub/expert_parallel_deepep_wrapper_test.cpp)
cpp_test(expert_parallel_deepep_wrapper_test SRCS ${expert_parallel_deepep_wrapper_test_SRCS} DEPS data_hub utils)
