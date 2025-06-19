# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE batch_scheduler_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/*.cpp)
list(FILTER batch_scheduler_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "batch_scheduler_SRCS: ${batch_scheduler_SRCS}")

add_library(batch_scheduler STATIC ${batch_scheduler_SRCS})

if(WITH_VLLM_FLASH_ATTN)
  set(MODELS_TEST_LIBS ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY})
else()
  set(MODELS_TEST_LIBS "")
endif()
set(MODELS_TEST_DEPS batch_scheduler runtime data_hub)

cpp_test(batch_scheduler_helper_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/batch_scheduler_test_helper_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(batch_scheduler_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/batch_scheduler_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(batch_scheduler_balance_reqs_algo_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/batch_scheduler_balance_reqs_algo_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(continuous_batching_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler/strategy/continuous_batching_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
cpp_test(pp_multibatch_wb_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/batch_scheduler//workload_balance/pp_multibatch_balancer_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
