#Copyright 2025 Tencent Inc.All rights reserved.
#Communicator module CMake configuration

file(GLOB COMMUNICATOR_SRCS *.cpp zmq/*.cpp nvida/*.cpp)
list(FILTER COMMUNICATOR_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER COMMUNICATOR_SRCS EXCLUDE REGEX ".*mock_.*\.cpp")
file(GLOB COMMUNICATOR_HDRS *.h zmq/*.h nvida/*.h)
list(FILTER COMMUNICATOR_HDRS EXCLUDE REGEX ".*test.h")
list(FILTER COMMUNICATOR_HDRS EXCLUDE REGEX ".*mock_.*\.h.*")

set(COMMUNICATOR_SOURCES ${COMMUNICATOR_SRCS} PARENT_SCOPE)
set(COMMUNICATOR_HEADERS ${COMMUNICATOR_HDRS} PARENT_SCOPE)

# Note: Mock libraries are not needed as tests use in-file mock functions
# with dlsym(RTLD_NEXT, ...) for API interception.
# Mock control is via testing_internal::EnableNcclMock(true/false)

message(STATUS "Communicator source files: ${COMMUNICATOR_SRCS}")