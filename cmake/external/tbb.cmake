# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
include(ExternalProject)

set(TBB_CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DTBB_TEST=OFF
    -DTBB_EXAMPLES=OFF
    -DTBBMALLOC_BUILD=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/tbb_install
    -DCMAKE_INSTALL_LIBDIR=lib64
    -DCMAKE_CXX_FLAGS=-Wno-stringop-overflow\ -Wno-error
)

execute_process(
    COMMAND bash -c "lscpu | grep 'Vendor ID' | grep -i 'AuthenticAMD'"
    RESULT_VARIABLE IS_AMD_CPU
    OUTPUT_QUIET
    ERROR_QUIET
)

if(IS_AMD_CPU EQUAL 0)
    message(STATUS "AMD CPU detected, disabling IPO for TBB.")
    list(APPEND TBB_CMAKE_ARGS -DTBB_ENABLE_IPO=OFF)
else()
    message(STATUS "Non-AMD CPU detected, using default TBB settings.")
endif()

ExternalProject_Add(tbb_external
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2022.2.0
    CMAKE_ARGS ${TBB_CMAKE_ARGS}
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/tbb_install/lib64/libtbb.so.12
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tbb_install/include)

add_library(TBB::tbb SHARED IMPORTED)
set_target_properties(TBB::tbb PROPERTIES
    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/tbb_install/lib64/libtbb.so.12
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/tbb_install/include
)

add_dependencies(TBB::tbb tbb_external)