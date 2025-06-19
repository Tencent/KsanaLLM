# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(FMT_INSTALL_DIR ${THIRD_PARTY_PATH}/download/fmt)

if(NOT DEFINED FMT_GIT_TAG)
    set(FMT_GIT_TAG 9.1.0)
endif()
set(FMT_GIT_URL  https://github.com/fmtlib/fmt/archive/${FMT_GIT_TAG}.tar.gz)

FetchContent_Declare(
    fmt
    URL        ${FMT_GIT_URL}
    SOURCE_DIR ${FMT_INSTALL_DIR}
)

FetchContent_GetProperties(fmt)
if(NOT fmt_POPULATED)
    FetchContent_Populate(fmt)

    add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR})
endif()

message(STATUS "Fmt source directory: ${fmt_SOURCE_DIR}")
message(STATUS "Fmt binary directory: ${fmt_BINARY_DIR}")

include_directories(${fmt_SOURCE_DIR}/include)
