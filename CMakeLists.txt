# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Dynamically detect GCC version and set the compiler path for compilation
execute_process(COMMAND /usr/bin/gcc --version OUTPUT_VARIABLE GCC_VERSION_USR_BIN ERROR_QUIET)
execute_process(COMMAND /usr/local/bin/gcc --version OUTPUT_VARIABLE GCC_VERSION_USR_LOCAL ERROR_QUIET)

# Extract version numbers
string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+" GCC_VERSION_USR_BIN "${GCC_VERSION_USR_BIN}")
string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+" GCC_VERSION_USR_LOCAL "${GCC_VERSION_USR_LOCAL}")

# Check if default GCC version is less than 11 (requires GCC 11+)
string(REGEX MATCH "^[0-9]+" GCC_MAJOR_VERSION_USR_BIN "${GCC_VERSION_USR_BIN}")
if(GCC_MAJOR_VERSION_USR_BIN LESS 11)
    if(GCC_VERSION_USR_LOCAL GREATER GCC_VERSION_USR_BIN)
        set(CMAKE_C_COMPILER /usr/local/bin/gcc)
        set(CMAKE_CXX_COMPILER /usr/local/bin/g++)
        message(STATUS "Using newer GCC from /usr/local/bin (version ${GCC_VERSION_USR_LOCAL})")
    else()
        set(CMAKE_C_COMPILER /usr/bin/gcc)
        set(CMAKE_CXX_COMPILER /usr/bin/g++)
        message(STATUS "Using system GCC from /usr/bin (version ${GCC_VERSION_USR_BIN})")
    endif()
else()
    message(STATUS "Using system GCC (version ${GCC_VERSION_USR_BIN}) - meets minimum requirements")
endif()

message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")

# support NVIDIA GPU
option(WITH_CUDA "Enable CUDA" ON)

# support Huawei NPU
option(WITH_ACL "Enable Ascend" OFF)

# support Tencent ZiXiao
option(WITH_TOPS "Enable Tops" OFF)

option(WITH_TESTING "Enable testing" OFF)
option(WITH_STANDALONE_TEST "Enable standalone testing" OFF)
option(WITH_EVENT_RECORD "Enable event record" OFF)
option(WITH_CLEAR_CACHE "Enable clear prefix cache" OFF)
option(WITH_LIGHTLY_CI_TEST "Enable lightly ci test" ON)
option(WITH_BOOST "Enable Boost" ON)
option(WITH_FLASH_ATTN_CACHE "Enable block table-based KV cache management for Flash Attention" ON)
  

# Enable use of Tencent internal libraries, for example, Tencent nameserver Polaris
option(WITH_INTERNAL_LIBRARIES "Enable use of Tencent internal libraries" OFF)
if(WITH_INTERNAL_LIBRARIES)
  add_compile_definitions(WITH_INTERNAL_LIBRARIES)
  message(STATUS "WITH_INTERNAL_LIBRARIES is ON: Tencent internal libraries (e.g. Polaris) will be used.")
else()
  message(STATUS "WITH_INTERNAL_LIBRARIES is OFF: Tencent internal libraries (e.g. Polaris) will NOT be used.")
endif()

if(WITH_CLEAR_CACHE)
  add_definitions("-DCLEAR_CACHE")
endif()

# Build external libraries statically by default
set(BUILD_SHARED_LIBS OFF)

if(WITH_CUDA)
  # support NVIDIA GPU
  project(ksana_llm LANGUAGES CXX CUDA)
elseif(WITH_ACL)
  # support Huawei NPU
  set(ASCEND_PRODUCT_TYPE "ascend910")
  set(ASCEND_PLATFORM_NAME "Ascend910B2C")
  set(ASCEND_CORE_TYPE "AiCore")
  set(ASCEND_RUN_MODE "ONBOARD")
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module)
  list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})

  # NOTE(karlluo): languages cce need configured cmake module before
  project(ksana_llm LANGUAGES CCE CXX)

elseif(WITH_TOPS)
  # support Tencent ZiXiao
  set(ARCH "gcu300")
  set(CMAKE_TOPS_COMPILER_TOOLKIT_ROOT "/opt/tops")
  list(APPEND CMAKE_MODULE_PATH "/opt/tops/cmake_modules")
  include(FindTOPS)
  project(ksana_llm LANGUAGES CXX TOPS)
else()
  message(FATAL_ERROR "Support platform is not selected. Select NVIDIA GPU with -DWITH_CUDA=ON, Huawei NPU with -DWITH_ACL=ON and Tencent ZiXiao with -DWITH_TOPS=ON")
endif()

if(NOT WITH_CUDA AND NOT WITH_ACL AND NOT WITH_TOPS)
  message(FATAL_ERROR "WITH_CUDA=OFF and WITH_ACL=OFF and WITH_TOPS=OFF is not allow")
endif()

set(USE_CXX11_ABI, "False")

# prepare ABI version
execute_process(COMMAND python "-c" "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)

if(NOT _PYTHON_SUCCESS EQUAL 0)
  message(FATAL_ERROR "run python -c \"import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');\" failed.")
endif()

# set compiler flags
if("${USE_CXX11_ABI}" STREQUAL "True")
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
  message(STATUS "Compile with CXX11 ABI")
else()
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  message(STATUS "Compile without CXX11 ABI")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third_party)

# Set TritonKernel Path
set(TRITON_KERNEL_DIRNAME "triton_kernel_files")
if(DEFINED ENV{KSANA_TRITON_KERNEL_PATH})
  set(TRITON_KERNEL_PATH "$ENV{KSANA_TRITON_KERNEL_PATH}/${TRITON_KERNEL_DIRNAME}" CACHE PATH "Path to TritonKernel files")
  message(STATUS "Using KSANA_TRITON_KERNEL_PATH from environment: ${TRITON_KERNEL_PATH}")
else()
  if(DEFINED ENV{HOME})
    set(TRITON_KERNEL_PATH "$ENV{HOME}/.cache/KsanaLLM/${TRITON_KERNEL_DIRNAME}" CACHE PATH "Path to TritonKernel files")
  else()
    set(TRITON_KERNEL_PATH "${CMAKE_BINARY_DIR}/${TRITON_KERNEL_DIRNAME}" CACHE PATH "Path to TritonKernel files")
    message(WARNING "Cannot get HOME environment variable, using build directory: ${TRITON_KERNEL_PATH}")
  endif()
endif()
message(STATUS "Setting TRITON_KERNEL_PATH: ${TRITON_KERNEL_PATH}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TRITON_KERNEL_PATH})

find_package(Git REQUIRED)

if(WITH_CUDA)
  # dedicate for Nvidia GPU
  option(CUDA_PTX_VERBOSE_INFO "build nvidia kernels with detailed ptx info" OFF)
  find_package(CUDA 11.8 REQUIRED)
  find_package(NCCL REQUIRED)

  if(NOT DEFINED SM OR "${SM}" STREQUAL "")
    message(STATUS "finding sm with ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py")
    execute_process(COMMAND python ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py OUTPUT_VARIABLE SM OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Auto detect SM is ${SM}")
  endif()

  if(WITH_FLASH_ATTN_CACHE)
    # Enable block table-based KV cache management.
    # This is supported by all Flash Attention implementations except official Flash Attention versions < 2.6.
    # Block table approach provides better memory efficiency and supports variable sequence lengths.
    add_definitions("-DENABLE_FLASH_ATTN_WITH_CACHE")
    message(STATUS "Flash Attention with block table-based KV cache: ENABLED")
  else()
    message(STATUS "Flash Attention with block table-based KV cache: DISABLED")
  endif()
  # 根据SM变量选择合适的CUDA架构
  if(NOT DEFINED SM)
    set(SM "90a")  # 默认使用90a
  endif()
  # 检查是否为Hopper架构（SM90+），Flash Attention 3需要Hopper架构支持
  string(SUBSTRING "${SM}" 0 1 SM_FIRST_CHAR)
  if(SM_FIRST_CHAR STREQUAL "9")
    message(STATUS "Hopper architecture (SM${SM}) detected, enabling Flash Attention 3...")
    include(FlashAttention3AutoBuild)
    check_flash_attn_3_available(FA3_AVAILABLE)
    if(NOT FA3_AVAILABLE)
      message(STATUS "Flash Attention 3 not found, setting up auto build...")
      auto_build_flash_attn_3()
    endif()
  else()
    message(STATUS "Flash Attention 3 disabled for SM${SM} architecture (requires Hopper SM90+)")
  endif()
endif()

if(WITH_BOOST)
  include(external/boost)
endif()

include(LLM_kernels)
include(base)

# set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -DWMMA -gdwarf-4 -gstrict-dwarf")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0 -gdwarf-4 -gstrict-dwarf")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -DWMMA -gdwarf-4 -gstrict-dwarf")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O3 -Werror=return-type -Wall -Wno-strict-aliasing -Wno-pointer-arith -Wno-ignored-attributes -Wno-deprecated -finline-functions -Wno-unknown-pragmas -Wno-pointer-arith -Wno-attributes -Wabi-tag")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -gdwarf-4 -gstrict-dwarf")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -gdwarf-4 -gstrict-dwarf")

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH $ORIGIN)

if(WITH_TESTING)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

message(STATUS "Build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

if(WITH_CUDA)
  include(nvidia)
endif()

if(WITH_ACL)
  include(ascend)
endif()

if(WITH_TOPS)
  include(zixiao)
endif()

# set cmake output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# set include headers
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)

# set headers
include_directories(
  ${COMMON_HEADER_DIRS}
  ${CUDA_INC_DIRS}
  ${ACL_INC_DIRS}
  ${TOPS_INC_DIRS}
)

# set linked libraries
link_directories(
  ${CUDA_LIB_DIRS}
  ${ACL_LIB_DIRS}
  ${TOPS_LIB_DIRS}
)

if(WITH_TESTING)
  enable_testing()
  include(external/gtest)
endif()

if(WITH_EVENT_RECORD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RECORD_EVENT")
endif()

# NOTE(karlluo): for PyTorch 2.7.1, torch/share/cmake/Caffe2/public/cuda.cmake's nvtx path depends on
# variable USE_SYSTEM_NVTX, we set it on to use system nvtx.
set(USE_SYSTEM_NVTX ON)

# include torch
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_VERSION)

if(TORCH_VERSION VERSION_LESS "1.5.0")
  message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
endif()
if(TORCH_VERSION VERSION_GREATER_EQUAL "2.2.0")
    add_definitions("-DENABLE_FP8_TORCH")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Torch config Error.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PY_INCLUDE_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Python config Error.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE USE_CXX11_ABI)
include_directories(${PY_INCLUDE_DIR})
include_directories(${TORCH_DIR}/include/torch/csrc/api/include/)
include_directories(${TORCH_DIR}/include/)
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python PATHS "${TORCH_DIR}/lib" NO_DEFAULT_PATH)
find_package(PythonLibs REQUIRED)

# NOTE(karlluo): for NPU environment openssl conflict with conda's ssl, take conda's ssl as libs
if(WITH_ACL AND WITH_TRPC_ENDPOINT)
  set(PY_LIB_DIR "")
  get_filename_component(PY_LIB_DIR ${PYTHON_LIBRARY} DIRECTORY)
  message(STATUS "WITH_TRPC_ENDPOINT using PY_LIB_DIR: " ${PY_LIB_DIR})
  link_directories(${PY_LIB_DIR})
endif()

include_directories(${PYTHON_INCLUDE_DIRS})
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(3rdparty)
add_subdirectory(examples)
add_subdirectory(src)
