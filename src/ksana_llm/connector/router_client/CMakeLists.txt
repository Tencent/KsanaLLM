# Copyright 2025 Tencent Inc. All rights reserved.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

# 收集 router_client 模块源文件
file(GLOB ROUTER_CLIENT_SRCS *.cpp)
list(FILTER ROUTER_CLIENT_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB ROUTER_CLIENT_HDRS *.h)
list(FILTER ROUTER_CLIENT_HDRS EXCLUDE REGEX ".*test.h")

# 将 router_client 源文件添加到父级变量中
set(ROUTER_CLIENT_SOURCES ${ROUTER_CLIENT_SRCS} PARENT_SCOPE)
set(ROUTER_CLIENT_HEADERS ${ROUTER_CLIENT_HDRS} PARENT_SCOPE)

# 输出调试信息
message(STATUS "RouterClient source files: ${ROUTER_CLIENT_SRCS}")