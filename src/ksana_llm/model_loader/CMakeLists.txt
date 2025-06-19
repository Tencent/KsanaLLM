# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE model_loader_SRCS
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/config_parser/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/file_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/model_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/weight_loader/*.cpp)

list(FILTER model_loader_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "model_loader_SRCS: ${model_loader_SRCS}")

add_library(model_loader STATIC ${model_loader_SRCS})
target_link_libraries(model_loader PUBLIC utils models)

if(NOT WITH_LIGHTLY_CI_TEST)
  # model_loader_test
  file(GLOB_RECURSE model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/model_loader_test.cpp)
  message(STATUS "model_loader_test_SRCS: ${model_loader_test_SRCS}")
  cpp_test(model_loader_test SRCS ${model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

  # llama_model_loader_test
  file(GLOB_RECURSE llama_model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/llama_model_loader_test.cpp)
  message(STATUS "llama_model_loader_test_SRCS: ${llama_model_loader_test_SRCS}")
  cpp_test(llama_model_loader_test SRCS ${llama_model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

  # new_deepseek_v3_model_loader_test
  file(GLOB_RECURSE new_deepseek_v3_model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/model_loader/new_deepseek_v3_model_loader_test.cpp)
  message(STATUS "new_deepseek_v3_model_loader_test_SRCS: ${new_deepseek_v3_model_loader_test_SRCS}")
  cpp_test(new_deepseek_v3_model_loader_test SRCS ${new_deepseek_v3_model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()
