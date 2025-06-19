#Copyright 2025 Tencent Inc.All rights reserved.
#
#== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

#初始化变量
set(CONNECTOR_SRCS "")
set(CONNECTOR_HDRS "")

#添加子目录
add_subdirectory(coordinator)
add_subdirectory(communicator)
add_subdirectory(router_client)

#收集顶层 connector 目录的源文件
file(GLOB BASE_CONNECTOR_SRCS *.cpp)
list(FILTER BASE_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB BASE_CONNECTOR_HDRS *.h)
list(FILTER BASE_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")

#收集 pipe / utils 子目录源文件
file(GLOB_RECURSE PIPE_CONNECTOR_SRCS pipe/*.cpp)
list(FILTER PIPE_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB_RECURSE UTILS_CONNECTOR_SRCS utils/*.cpp)
list(FILTER UTILS_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB_RECURSE PIPE_CONNECTOR_HDRS pipe/*.h)
list(FILTER PIPE_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")
file(GLOB_RECURSE UTILS_CONNECTOR_HDRS utils/*.h)
list(FILTER UTILS_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")

# 合并所有源文件
list(APPEND CONNECTOR_SRCS
    ${BASE_CONNECTOR_SRCS}
    ${COORDINATOR_SOURCES}
    ${COMMUNICATOR_SOURCES}
    ${ROUTER_CLIENT_SOURCES}
    ${PIPE_CONNECTOR_SRCS}
    ${UTILS_CONNECTOR_SRCS}
)
list(APPEND CONNECTOR_HDRS
    ${BASE_CONNECTOR_HDRS}
    ${COORDINATOR_HEADERS}
    ${COMMUNICATOR_HEADERS}
    ${ROUTER_CLIENT_HEADERS}
    ${PIPE_CONNECTOR_HDRS}
    ${UTILS_CONNECTOR_HDRS}
)

# 只有在 WITH_CUDA 开启时才包含 NCCL 相关文件（communicator 目录已自动处理）
if(WITH_CUDA)
  message(STATUS "CUDA enabled, including NCCL communicator files")
else()
  message(STATUS "CUDA not enabled, excluding NCCL communicator files")
endif()

message(STATUS "CONNECTOR_SRCS: ${CONNECTOR_SRCS}")
message(STATUS "CONNECTOR_HDRS: ${CONNECTOR_HDRS}")
message(STATUS "CPPZMQ_INCLUDE_DIR: ${CPPZMQ_INCLUDE_DIR}")
message(STATUS "LIBZMQ_INCLUDE_DIR: ${LIBZMQ_INCLUDE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# Create the connector library
add_library(connector STATIC ${CONNECTOR_SRCS})

add_dependencies(connector cppzmq libzmq)

target_include_directories(connector
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBZMQ_INCLUDE_DIR}
        ${CPPZMQ_INCLUDE_DIR}
)

target_link_libraries(connector PUBLIC
    utils
    cppzmq
    libzmq
)

if(WITH_CUDA)
    target_link_libraries(connector PUBLIC ${NCCL_LIBRARIES})
endif()

target_link_libraries(connector PUBLIC ${CURL_LIBRARIES})

get_target_property(_inc_dirs connector INCLUDE_DIRECTORIES)
message(STATUS "connector include dirs: ${_inc_dirs}")

# Note: Mock libraries are not needed as tests use in-file mock functions
# with dlsym(RTLD_NEXT, ...) for API interception.
# Mock control is via testing_internal::EnableNcclMock(true/false) and
# testing_internal::EnableZmqMock(true/false)

# Add testing if enabled
if(WITH_TESTING)
    file(GLOB_RECURSE TEST_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/connector/*test.cpp)
    add_executable(connector_test ${TEST_SRCS})
    set_target_properties(connector_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_compile_options(connector_test PRIVATE -fno-access-control -g -O0)
    
    # Note: No dependencies on mock libraries needed as tests use internal mock control
    
    if(WITH_CUDA)
        target_link_libraries(connector_test
            connector
            gtest
            runtime
            gtest_main
            gmock
            ${CURL_LIBRARIES}
            cuda
        )
    else()
        target_link_libraries(connector_test
            connector
            gtest
            gtest_main
            gmock
            runtime
            ${CURL_LIBRARIES}
        )
    endif()
    add_test(NAME connector_test COMMAND connector_test)
endif()