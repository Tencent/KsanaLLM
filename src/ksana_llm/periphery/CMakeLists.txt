cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(periphery_SRCS,"")
file(GLOB_RECURSE periphery_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/periphery/*.cpp)
list(FILTER periphery_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "periphery_SRCS: ${periphery_SRCS}")

# Get Git version information
execute_process(
    COMMAND git describe --tags --always
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "GIT_VERSION: ${GIT_VERSION}")
# Get Git commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "GIT_COMMIT_HASH: ${GIT_COMMIT_HASH}")

# Get Git branch name
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "GIT_BRANCH: ${GIT_BRANCH}")

# Configure version_reporter.h file with version information
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version_info.h.in
    ${CMAKE_BINARY_DIR}/include/ksana_llm/periphery/version_info.h
    @ONLY
)
message(STATUS "Configuring version_info.h at CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/version_info.h")
message(STATUS "Configuring version_info.h at CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/version_info.h")

include_directories(${CMAKE_BINARY_DIR}/include)

# 查找 libcurl
find_package(CURL REQUIRED)

add_library(periphery STATIC ${periphery_SRCS})
target_link_libraries(periphery PUBLIC -lpthread utils CURL::libcurl)

# 如果启用了独立测试
if(WITH_STANDALONE_TEST)
  # 查找测试源文件
  file(GLOB_RECURSE periphery_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/periphery/*test.cpp)
  message(STATUS "periphery_test_SRCS: ${periphery_test_SRCS}")

  # 启用测试
  enable_testing()

  # 创建测试可执行文件
  add_executable(periphery_test ${periphery_test_SRCS})
  target_link_libraries(periphery_test
    periphery
    gtest_main
    gtest
    gmock_main
    gmock
    -pthread)

  # 添加测试
  add_test(NAME periphery_test
    COMMAND periphery_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  message(STATUS "Added test: periphery_test")
endif()
