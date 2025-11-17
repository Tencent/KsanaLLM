# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE torch_op_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/torch_op/*.cpp)
list(FILTER torch_op_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "torch_op_SRCS: ${torch_op_SRCS}")

add_library(torch_serving SHARED ${torch_op_SRCS} ${host_stub_SRCS})

set_target_properties(torch_serving PROPERTIES
    BUILD_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH OFF
)

if(WITH_CUDA OR WITH_TOPS)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES} ksana_llm_libs)
endif()

if(WITH_ACL)
  target_link_libraries(torch_serving PRIVATE ${TORCH_LIBRARIES} ksana_llm_libs -Wl,--whole-archive layers -Wl,--no-whole-archive kernels)
endif()

# for test
if(WITH_STANDALONE_TEST)
  file(GLOB_RECURSE serving_op_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/torch_op/*test.cpp)
  message(STATUS "serving_op_test_SRCS: ${serving_op_test_SRCS}")
  cpp_test(serving_op_test SRCS ${serving_op_test_SRCS} DEPS torch_serving runtime utils data_hub)
endif()
