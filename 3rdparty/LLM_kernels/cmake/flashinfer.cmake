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
    # Performance optimization: customize flashinfer build to compile only batch_paged_prefill kernels
    # Reduce compilation time by 95% and minimizing binary size
    file(READ "${FLASHINFER_INSTALL_DIR}/CMakeLists.txt" FLASHINFER_CMAKE_CONTENT)

    # Disable the compilation of decode_kernels
    string(REPLACE
        "add_library(decode_kernels STATIC \${DECODE_KERNELS_SRCS})"
        "# add_library(decode_kernels STATIC \${DECODE_KERNELS_SRCS}) # Disabled by parent repo ksana"
        FLASHINFER_CMAKE_CONTENT "${FLASHINFER_CMAKE_CONTENT}")

    string(REPLACE
        "target_include_directories(decode_kernels PRIVATE \${FLASHINFER_INCLUDE_DIR})"
        "# target_include_directories(decode_kernels PRIVATE \${FLASHINFER_INCLUDE_DIR})"
        FLASHINFER_CMAKE_CONTENT "${FLASHINFER_CMAKE_CONTENT}")

    string(REPLACE
        "target_link_libraries(decode_kernels PRIVATE Boost::math)"
        "# target_link_libraries(decode_kernels PRIVATE Boost::math)"
        FLASHINFER_CMAKE_CONTENT "${FLASHINFER_CMAKE_CONTENT}")

    # Only compile batch_paged_prefill kernels
    string(REPLACE
        "file(GLOB_RECURSE PREFILL_KERNELS_SRCS\n     \${PROJECT_SOURCE_DIR}/src/generated/*prefill_head*.cu)"
        "file(GLOB PREFILL_KERNELS_SRCS\n     \${PROJECT_SOURCE_DIR}/src/generated/batch_paged_prefill_head*.cu)"
        FLASHINFER_CMAKE_CONTENT "${FLASHINFER_CMAKE_CONTENT}")

    file(WRITE "${FLASHINFER_INSTALL_DIR}/CMakeLists.txt" "${FLASHINFER_CMAKE_CONTENT}")

    message(STATUS "Modified flashinfer: only compile batch_paged_prefill kernels")
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
    set(FLASHINFER_GEN_POS_ENCODING_MODES "0" CACHE STRING "")

    # Save current compile flags and suppress warnings for flashinfer subproject
    set(FLASHINFER_SAVED_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(FLASHINFER_SAVED_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    # Append -Wno-sign-compare for c++ compile flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    # Append --compiler-options -Wno-sign-compare for cuda compile flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wno-sign-compare")

    add_subdirectory(${repo-flashinfer_SOURCE_DIR} ${repo-flashinfer_BINARY_DIR})

    # Restore original compile flags
    set(CMAKE_CXX_FLAGS "${FLASHINFER_SAVED_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${FLASHINFER_SAVED_CUDA_FLAGS}")
endif()

include_directories(${repo-flashinfer_SOURCE_DIR}/include 
                    ${repo-flashinfer_SOURCE_DIR}/csrc
                    ${repo-flashinfer_SOURCE_DIR}/src/generated)