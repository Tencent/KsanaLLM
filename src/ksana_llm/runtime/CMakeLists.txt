# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

# 为 LayerProgressTracker 添加独立的编译选项
add_library(layer_progress_tracker STATIC
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/layer_progress_tracker.cpp)

target_link_libraries(layer_progress_tracker PUBLIC
  -lpthread utils gflags)

if(WITH_CUDA)
  cpp_test(layer_progress_tracker_test 
    SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/layer_progress_tracker_test.cpp 
    DEPS layer_progress_tracker)
endif()

# 排除 LayerProgressTracker 文件，不编译到 runtime 库中
file(GLOB_RECURSE runtime_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/*.cpp)
list(FILTER runtime_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER runtime_SRCS EXCLUDE REGEX ".*layer_progress_tracker.cpp")
message(STATUS "runtime_SRCS: ${runtime_SRCS}")

set(runtime_nvidia_LIBS "")

if(WITH_CUDA)
  list(APPEND runtime_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublasLt -lcublas)
endif()

set(runtime_ascend_LIBS "")

if(WITH_ACL)
  list(APPEND runtime_ascend_LIBS ${ACL_SHARED_LIBS})
  list(FILTER runtime_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
endif()

set(runtime_zixiao_LIBS "")

if(WITH_TOPS)
  list(APPEND runtime_ascend_LIBS ${TOPS_SHARED_LIBS})
  list(FILTER runtime_SRCS EXCLUDE REGEX ".*cuda_graph_runner.cpp")
endif()

add_library(runtime STATIC ${runtime_SRCS})
add_dependencies(runtime utils fmt gflags layer_progress_tracker multi_batch_controller)
target_link_libraries(runtime PUBLIC
  -lpthread -ldl gflags ${runtime_nvidia_LIBS} ${runtime_ascend_LIBS}
  model_loader layer_progress_tracker models utils transfer multi_batch_controller
)

# for test
file(GLOB_RECURSE runtime_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/threadpool_test.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/draft_generator/trie_generator_test.cpp)
if(WITH_CUDA AND DEFINED SM AND "${SM}" STREQUAL "90a")
  list(APPEND runtime_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/runtime/weight_instance_test.cpp)
endif()
message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
cpp_test(runtime_test SRCS ${runtime_test_SRCS} DEPS runtime)
