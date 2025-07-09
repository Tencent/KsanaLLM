# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE utils_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/*.cpp)
list(FILTER utils_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/nvidia/.*.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/ascend/.*.cpp")
list(FILTER utils_SRCS EXCLUDE REGEX ".*/zixiao/.*.cpp")
message(STATUS "utils_SRCS: ${utils_SRCS}")

set(utils_nvidia_SRCS, "")
set(utils_nvidia_LIBS, "")

if(WITH_CUDA)
  file(GLOB_RECURSE utils_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/nvidia/*.cpp)
  list(APPEND utils_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt -lcuda llm_kernels_nvidia_utils)
endif()

set(utils_ascend_SRCS, "")
set(utils_ascend_LIBS, "")

if(WITH_ACL)
  file(GLOB_RECURSE utils_ascend_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/ascend/*.cpp)
  list(APPEND utils_ascend_LIBS ${ACL_SHARED_LIBS} hccl llm_kernels_ascend_utils_common)
endif()

set(utils_zixiao_SRCS, "")
set(utils_zixiao_LIBS, "")

if(WITH_TOPS)
  file(GLOB_RECURSE utils_zixiao_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/zixiao/*.cpp)
  list(APPEND utils_zixiao_LIBS ${ZIXIAO_SHARED_LIBS} topsrt llm_kernels_zixiao_utils_common)
endif()

add_library(utils STATIC ${utils_SRCS} ${utils_nvidia_SRCS} ${utils_ascend_SRCS} ${utils_zixiao_SRCS})
add_dependencies(utils fmt gflags)
target_link_libraries(utils PUBLIC
  -lpthread -ldl gflags loguru yaml-cpp profiler ${utils_nvidia_LIBS} ${utils_ascend_LIBS} ${utils_zixiao_LIBS}
  absl::base absl::strings re2 ${TORCH_LIBRARIES} ${PYTHON_LIBRARY}
)

# for test
if(WITH_STANDALONE_TEST)
  file(GLOB_RECURSE utils_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/*test.cpp
    ${PROJECT_SOURCE_DIR}/tests/test.cpp)
  list(FILTER utils_test_SRCS EXCLUDE REGEX ".*environment_test.cpp")
  message(STATUS "utils_test_SRCS: ${utils_test_SRCS}")
  cpp_test(utils_test SRCS ${utils_test_SRCS} DEPS runtime utils)
  cpp_test(environment_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/utils/environment_test.cpp DEPS runtime utils)
endif()
