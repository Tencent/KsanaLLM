# Copyright 2025 Tencent Inc. All rights reserved.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

# 收集 coordinator 模块源文件
file(GLOB COORDINATOR_SRCS *.cpp)
list(FILTER COORDINATOR_SRCS EXCLUDE REGEX ".*test.cpp")
file(GLOB COORDINATOR_HDRS *.h)
list(FILTER COORDINATOR_HDRS EXCLUDE REGEX ".*test.h")

# 将 coordinator 源文件添加到父级变量中
set(COORDINATOR_SOURCES ${COORDINATOR_SRCS} PARENT_SCOPE)
set(COORDINATOR_HEADERS ${COORDINATOR_HDRS} PARENT_SCOPE)

# 输出调试信息
message(STATUS "Coordinator source files: ${COORDINATOR_SRCS}")
