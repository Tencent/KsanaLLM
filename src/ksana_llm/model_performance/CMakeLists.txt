# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE model_performance_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/perf_profile_config_builder_for_csv.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/model_performance_runner.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/communication_performance_runner.cpp
  )

list(FILTER model_performance_SRCS EXCLUDE REGEX ".*test.cpp")

add_library(model_performance STATIC ${model_performance_SRCS})
target_link_libraries(model_performance PUBLIC cache_manager models runtime layers samplers runtime ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS} data_hub distributed)


add_executable(run_model_performance ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/run_model_performance.cpp)
target_link_libraries(run_model_performance PUBLIC model_performance yaml-cpp)

add_executable(run_communcation_performance ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/run_communication_performance.cpp)
target_link_libraries(run_communcation_performance PUBLIC model_performance)


# for test
if(WITH_CUDA)
  file(GLOB_RECURSE model_performance_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_performance/*test.cpp)
  cpp_test(model_performance_test SRCS ${model_performance_test_SRCS} DEPS model_performance)
endif()
