# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)


# 设置 cppzmq 的 header-only 路径
set(CPPZMQ_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/cppzmq)

FetchContent_Populate(download_cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.10.0
    SOURCE_DIR ${CPPZMQ_INCLUDE_DIR}
)

# 创建 cppzmq interface target
add_library(cppzmq INTERFACE)
target_include_directories(cppzmq INTERFACE ${CPPZMQ_INCLUDE_DIR} ${LIBZMQ_INCLUDE_DIR})
target_link_libraries(cppzmq INTERFACE libzmq)

message(STATUS "cppzmq headers configured at: ${CPPZMQ_INCLUDE_DIR}")
message(STATUS "cppzmq: THIRD_PARTY_PATH is ${THIRD_PARTY_PATH}")

set(CPPZMQ_FOUND TRUE CACHE BOOL "cppzmq found" FORCE)
