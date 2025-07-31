# Copyright 2025 Tencent Inc. All rights reserved.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

# 收集 router_client 模块源文件
file(GLOB ROUTER_CLIENT_SRCS *.cpp)
list(FILTER ROUTER_CLIENT_SRCS EXCLUDE REGEX ".*test.cpp")

# 排除 resolved_endpoint 的两个实现文件，稍后根据配置选择
list(FILTER ROUTER_CLIENT_SRCS EXCLUDE REGEX ".*resolved_endpoint_internal.cpp")
list(FILTER ROUTER_CLIENT_SRCS EXCLUDE REGEX ".*resolved_endpoint_simple.cpp")

# 根据 WITH_INTERNAL_LIBRARIES 选择合适的 resolved_endpoint 实现
if(WITH_INTERNAL_LIBRARIES)
    list(APPEND ROUTER_CLIENT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/resolved_endpoint_internal.cpp")
    message(STATUS "Using internal resolved_endpoint implementation with Polaris support")
else()
    list(APPEND ROUTER_CLIENT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/resolved_endpoint_simple.cpp")
    message(STATUS "Using simple resolved_endpoint implementation")
endif()

file(GLOB ROUTER_CLIENT_HDRS *.h)
list(FILTER ROUTER_CLIENT_HDRS EXCLUDE REGEX ".*test.h")

# 将 router_client 源文件添加到父级变量中
set(ROUTER_CLIENT_SOURCES ${ROUTER_CLIENT_SRCS} PARENT_SCOPE)
set(ROUTER_CLIENT_HEADERS ${ROUTER_CLIENT_HDRS} PARENT_SCOPE)

# 输出调试信息
message(STATUS "RouterClient source files: ${ROUTER_CLIENT_SRCS}")