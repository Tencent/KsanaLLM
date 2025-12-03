# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(FLASHINFER_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flashinfer)

# flashinfer
# using the same GIT_TAG as the one used in 
# https://github.com/sgl-project/sglang/blob/9858113c336f4565a0a35f9a990cdada0de1988f/sgl-kernel/CMakeLists.txt#L64
FetchContent_Declare(
    repo-flashinfer
    GIT_REPOSITORY https://github.com/flashinfer-ai/flashinfer.git
    GIT_TAG        9220fb3443b5a5d274f00ca5552f798e225239b7
    GIT_SHALLOW    OFF
    SOURCE_DIR ${FLASHINFER_INSTALL_DIR}
)

if(NOT repo-flashinfer_POPULATED)
    FetchContent_Populate(repo-flashinfer)
endif()

if(WITH_CUDA)
    set(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS "false" CACHE STRING "")
    set(FLASHINFER_DECODE OFF CACHE BOOL "")
    set(FLASHINFER_PREFILL OFF CACHE BOOL "")
    set(FLASHINFER_PAGE OFF CACHE BOOL "")
    set(FLASHINFER_CASCADE OFF CACHE BOOL "")
    set(FLASHINFER_SAMPLING OFF CACHE BOOL "")
    set(FLASHINFER_NORM OFF CACHE BOOL "")
    set(FLASHINFER_FASTDIV_TEST OFF CACHE BOOL "")
    set(FLASHINFER_FASTDEQUANT_TEST OFF CACHE BOOL "")
    set(FLASHINFER_DISTRIBUTED OFF CACHE BOOL "")

    add_subdirectory(${repo-flashinfer_SOURCE_DIR} ${repo-flashinfer_BINARY_DIR})
endif()

include_directories(${repo-flashinfer_SOURCE_DIR}/include 
                    ${repo-flashinfer_SOURCE_DIR}/csrc
                    ${repo-flashinfer_SOURCE_DIR}/src/generated)