# Copyright 2024 Tencent Inc.  All rights reserved.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# support NVIDIA GPU
option(WITH_CUDA "Enable CUDA" ON)

# support Huawei NPU
option(WITH_ACL "Enable Ascend" OFF)

# support Tencent ZiXiao
option(WITH_TOPS "Enable Tops" OFF)

option(WITH_TESTING "Enable testing" OFF)
option(CUDA_PTX_VERBOSE_INFO "build nvidia kernels with detailed ptx info" OFF)

if(WITH_CUDA)
  project(LLMKernels LANGUAGES CXX CUDA)
  # Use 16 threads to accelerate CUDA compilation
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --threads 16)
elseif(WITH_ACL)
  set(ASCEND_PRODUCT_TYPE "ascend910")
  set(ASCEND_CORE_TYPE "AiCore")
  set(ASCEND_RUN_MODE "ONBOARD")
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module)
  set(SOC_VERSION "ascend910b1" CACHE STRING "system on chip type")
  set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
  set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim/cpu")

  message(STATUS "SOC_VERSION: ${SOC_VERSION}")
  message(STATUS "RUN_MODE: ${RUN_MODE}")
  message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
  message(STATUS "CCE_CMAKE_PATH: ${CCE_CMAKE_PATH}")
  message(STATUS "ASCEND_INSTALL_PATH: ${ASCEND_INSTALL_PATH}")
  message(STATUS "ASCEND_RUN_MODE: ${ASCEND_RUN_MODE}")
  message(STATUS "ASCEND_CORE_TYPE: ${ASCEND_CORE_TYPE}")
  message(STATUS "ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

  if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
  endif()

  include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
  list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

  # NOTE(karlluo): languages cce need configured cmake module before
  project(LLMKernels LANGUAGES CCE CXX)
elseif(WITH_TOPS)
  set(ARCH "gcu300")
  project(LLMKernels LANGUAGES CXX)
else()
  message(FATAL_ERROR "Support platform is not selected. Select NVIDIA GPU with -DWITH_CUDA=ON, Huawei NPU with -DWITH_ACL=ON and Tencent ZiXiao with -DWITH_TOPS=ON")
endif()

find_package(Git QUIET)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD OUTPUT_VARIABLE COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)

link_directories(/usr/lib64)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR}/cmake)
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third_party)

# set compiler flags
set(CXX_STD "17" CACHE STRING "C++ standard")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DWMMA -gdwarf-4 -gstrict-dwarf")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0 -gdwarf-4 -gstrict-dwarf")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

if(WITH_TESTING)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# set cmake output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(base)
include(nvidia)
include(ascend)
include(flashinfer)

if(WITH_CUDA)
  include(fmt)
  include(yaml-cpp)
endif()

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch half")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/half
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/half failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

# set headers
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${CUDA_INC_DIRS}
  ${ACL_INC_DIRS}
)

# set linked libraries
link_directories(
  ${CUDA_LIB_DIRS}
  ${ACL_LIB_DIRS}
)

include(test)

# add subdirectory
add_subdirectory(csrc)
add_subdirectory(tests)

if(WITH_CUDA)
  add_subdirectory(tools/search_best_gemm_algo)
endif()
