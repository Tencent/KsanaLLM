#Copyright 2025 Tencent Inc.All rights reserved.
#
#== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

set(CONNECTOR_SRCS "")
set(CONNECTOR_HDRS "")

add_subdirectory(coordinator)
add_subdirectory(communicator)
add_subdirectory(router_client)

file(GLOB BASE_CONNECTOR_SRCS *.cpp)
list(FILTER BASE_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB BASE_CONNECTOR_HDRS *.h)
list(FILTER BASE_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")

list(APPEND CONNECTOR_SRCS
    ${BASE_CONNECTOR_SRCS}
    ${COORDINATOR_SOURCES}
    ${COMMUNICATOR_SOURCES}
    ${ROUTER_CLIENT_SOURCES}
)
list(APPEND CONNECTOR_HDRS
    ${BASE_CONNECTOR_HDRS}
    ${COORDINATOR_HEADERS}
    ${COMMUNICATOR_HEADERS}
    ${ROUTER_CLIENT_HEADERS}
)

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

target_link_libraries(connector PRIVATE utils cppzmq libzmq TBB::tbb ${CURL_LIBRARIES})

if(WITH_CUDA)
    target_link_libraries(connector PRIVATE ${NCCL_LIBRARIES})
endif()

if(WITH_INTERNAL_LIBRARIES)
    target_link_libraries(connector PRIVATE  polaris-cpp::polaris_api polaris-cpp::polaris_cl5_api)
    target_include_directories(connector PUBLIC ${POLARIS_CPP_SOURCE_DIR}/include)
endif()

# Add testing if enabled
if(WITH_TESTING)
    file(GLOB_RECURSE TEST_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/connector/*test.cpp)
    add_executable(connector_test ${TEST_SRCS})
    set_target_properties(connector_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    target_compile_options(connector_test PRIVATE -fno-access-control -g -O0)

    target_link_libraries(connector_test PRIVATE connector gtest runtime gtest_main gmock ${CURL_LIBRARIES} TBB::tbb)

    if(WITH_CUDA)
        target_link_libraries(connector_test PRIVATE cuda )
    endif()

    if(WITH_INTERNAL_LIBRARIES)
        target_link_libraries(connector_test PRIVATE polaris-cpp::polaris_api polaris-cpp::polaris_cl5_api)
    endif()
    add_test(NAME connector_test COMMAND connector_test)
endif()