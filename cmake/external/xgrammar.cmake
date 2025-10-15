# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

# Fetch xgrammar
set(XGRAMMAR_SOURCE_DIR ${THIRD_PARTY_PATH}/xgrammar)
FetchContent_Declare(xgrammar
    GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
    GIT_TAG v0.1.25
    SOURCE_DIR ${XGRAMMAR_SOURCE_DIR}
)
FetchContent_Populate(xgrammar)

# Fetch dlpack
set(DLPACK_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/dlpack)
FetchContent_Populate(dlpack
    GIT_REPOSITORY https://github.com/dmlc/dlpack.git
    GIT_TAG v1.1
    SOURCE_DIR ${DLPACK_INCLUDE_DIR}
)

# Use picojson from xgrammar's 3rdparty directory
set(PICOJSON_INCLUDE_DIR ${XGRAMMAR_SOURCE_DIR}/3rdparty/picojson)

# Collect core source files from xgrammar
file(GLOB XGRAMMAR_SOURCES 
    "${XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
    "${XGRAMMAR_SOURCE_DIR}/cpp/support/*.cc"
)

# Create static library
add_library(xgrammar STATIC ${XGRAMMAR_SOURCES})

# Set include directories
target_include_directories(xgrammar PUBLIC
    ${XGRAMMAR_SOURCE_DIR}/include
    ${DLPACK_INCLUDE_DIR}/include
)

# Private include directories for internal headers
target_include_directories(xgrammar PRIVATE
    ${XGRAMMAR_SOURCE_DIR}/cpp
    ${PICOJSON_INCLUDE_DIR}
)

target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)

if(DEFINED PYBIND11_INCLUDE_DIR OR TARGET pybind11::module)
    target_compile_definitions(xgrammar PUBLIC PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF)
endif()

# Create alias for easier consumption
add_library(xgrammar::xgrammar ALIAS xgrammar)

# Add include directories globally (following the pattern of other libraries like fmt and nlohmann_json)
include_directories(${XGRAMMAR_SOURCE_DIR}/include)
include_directories(${DLPACK_INCLUDE_DIR}/include)
