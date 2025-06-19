# Copyright 2024 Tencent Inc.  All rights reserved.

# ======================================================================

include(ExternalProject)

# Define paths
set(LIBZMQ_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libzmq)
set(LIBZMQ_INCLUDE_DIR ${LIBZMQ_INSTALL_DIR}/include)
set(LIBZMQ_LIB_DIR ${LIBZMQ_INSTALL_DIR}/lib)
set(LIBZMQ_STATIC_LIB ${LIBZMQ_LIB_DIR}/libzmq.a)

# External project to build libzmq
ExternalProject_Add(
    libzmq_external
    PREFIX ${THIRD_PARTY_PATH}/libzmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG v4.3.5
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBZMQ_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED=OFF
        -DBUILD_STATIC=ON
        -DZMQ_BUILD_TESTS=OFF
        -DWITH_PERF_TOOL=OFF
        -DWITH_DOCS=OFF
    BUILD_BYPRODUCTS ${LIBZMQ_STATIC_LIB}
)

# 确保include目录存在，避免CMake报错
file(MAKE_DIRECTORY ${LIBZMQ_INCLUDE_DIR})

# Create imported target for libzmq
add_library(libzmq STATIC IMPORTED GLOBAL)
add_dependencies(libzmq libzmq_external)

set_target_properties(libzmq PROPERTIES
    IMPORTED_LOCATION ${LIBZMQ_STATIC_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${LIBZMQ_INCLUDE_DIR}
)

# System dependencies for Unix-based platforms
if(UNIX AND NOT APPLE)
    set_property(TARGET libzmq PROPERTY INTERFACE_LINK_LIBRARIES rt pthread dl)
elseif(APPLE)
    set_property(TARGET libzmq PROPERTY INTERFACE_LINK_LIBRARIES pthread)
endif()

# Informative messages
message(STATUS "ZeroMQ include directory: ${LIBZMQ_INCLUDE_DIR}")
message(STATUS "ZeroMQ static library: ${LIBZMQ_STATIC_LIB}")