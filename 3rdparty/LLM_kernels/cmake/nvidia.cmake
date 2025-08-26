# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT WITH_CUDA)
  return()
endif()

find_package(CUDA 11.2 REQUIRED)

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()

if(NOT DEFINED SM)
  execute_process(COMMAND python ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py OUTPUT_VARIABLE SM OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/triton_kernel_files)

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch cutlass c3x")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/c3x/cutlass
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/c3x/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch cutlass c4x")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/c4x/cutlass
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/c4x/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

if(CUDA_PTX_VERBOSE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall --generate-line-info -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")

# set CUDA related
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
set(SM_SETS 80 86 89 90 90a)
set(IS_SUPPORT_DEEPGEMM "FALSE")

# check if custom define SM
if(NOT DEFINED SM)
  foreach(SM_NUM IN LISTS SM_SETS)
    if(SM_NUM VERSION_GREATER_EQUAL "90")
      set(IS_SUPPORT_DEEPGEMM "TRUE")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_NUM},code=sm_${SM_NUM}")
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
    message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
    string(REGEX MATCHALL "[0-9]" SUB_VER_NUM "${SM_NUM}")
    list(JOIN SUB_VER_NUM "." SM_ARCH_VER)
    list(APPEND TORCH_CUDA_ARCH_LIST ${SM_ARCH_VER})
  endforeach()
elseif("${SM}" MATCHES ",")
  # Multiple SM values
  string(REPLACE "," ";" SM_LIST ${SM})

  foreach(SM_NUM IN LISTS SM_LIST)
    if(SM_NUM VERSION_GREATER_EQUAL "90")
      set(IS_SUPPORT_DEEPGEMM "TRUE")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_NUM},code=sm_${SM_NUM}")
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
    string(REGEX MATCHALL "[0-9]" SUB_VER_NUM "${SM_NUM}")
    list(JOIN SUB_VER_NUM "." SM_ARCH_VER)
    list(APPEND TORCH_CUDA_ARCH_LIST ${SM_ARCH_VER})
  endforeach()
else()
  if(SM VERSION_GREATER_EQUAL "90")
    set(IS_SUPPORT_DEEPGEMM "TRUE")
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM},code=sm_${SM}")
  list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM})
  message(STATUS "Assign GPU architecture (sm=${SM})")
  message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
  string(REGEX MATCHALL "[0-9]" SUB_VER_NUM "${SM}")
  list(JOIN SUB_VER_NUM "." SM_ARCH_VER)
  list(APPEND TORCH_CUDA_ARCH_LIST ${SM_ARCH_VER})
endif()

# setting cutlass v3.x
add_library(cutlass_c3x INTERFACE)
set(CUTLASS_C3X_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/c3x/cutlass)
set(CUTLASS_C3X_HEADER_DIR ${CUTLASS_C3X_SOURCE_DIR}/include)
set(CUTLASS_C3X_TOOLS_HEADER_DIR ${CUTLASS_C3X_SOURCE_DIR}/tools/util/include)
target_include_directories(cutlass_c3x INTERFACE
    ${CUTLASS_C3X_HEADER_DIR}
    ${CUTLASS_C3X_TOOLS_HEADER_DIR}
)
subproject_version(${CUTLASS_C3X_SOURCE_DIR} CUTLASS_C3X_VERSION)
set(CUTLASS_C3X_VERSION_SUB_LIST ${CUTLASS_C3X_VERSION})
string(REPLACE "." ";" CUTLASS_C3X_VERSION_SUB_LIST "${CUTLASS_C3X_VERSION}")
message(STATUS "cutlass c3x version is: ${CUTLASS_C3X_VERSION}")
list(GET CUTLASS_C3X_VERSION_SUB_LIST 0 CUTLASS_C3X_MAJOR_VERSION)
list(GET CUTLASS_C3X_VERSION_SUB_LIST 1 CUTLASS_C3X_MINOR_VERSION)
list(GET CUTLASS_C3X_VERSION_SUB_LIST 2 CUTLASS_C3X_PATCH_VERSION)
target_compile_definitions(cutlass_c3x INTERFACE
    CUTLASS_MAJOR_VERSION=${CUTLASS_C3X_MAJOR_VERSION}
    CUTLASS_MINOR_VERSION=${CUTLASS_C3X_MINOR_VERSION}
    CUTLASS_PATCH_VERSION=${CUTLASS_C3X_PATCH_VERSION}
)

# setting cutlass v4.x
add_library(cutlass_c4x INTERFACE)
set(CUTLASS_C4X_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/c4x/cutlass)
set(CUTLASS_C4X_HEADER_DIR ${CUTLASS_C4X_SOURCE_DIR}/include)
set(CUTLASS_C4X_TOOLS_HEADER_DIR ${CUTLASS_C4X_SOURCE_DIR}/tools/util/include)
target_include_directories(cutlass_c4x INTERFACE
    ${CUTLASS_C4X_HEADER_DIR}
    ${CUTLASS_C4X_TOOLS_HEADER_DIR}
)
subproject_version(${CUTLASS_C4X_SOURCE_DIR} CUTLASS_C4X_VERSION)
set(CUTLASS_C4X_VERSION_SUB_LIST ${CUTLASS_C4X_VERSION})
string(REPLACE "." ";" CUTLASS_C4X_VERSION_SUB_LIST "${CUTLASS_C4X_VERSION}")
message(STATUS "cutlass c4x version is: ${CUTLASS_C4X_VERSION}")
list(GET CUTLASS_C4X_VERSION_SUB_LIST 0 CUTLASS_C4X_MAJOR_VERSION)
list(GET CUTLASS_C4X_VERSION_SUB_LIST 1 CUTLASS_C4X_MINOR_VERSION)
list(GET CUTLASS_C4X_VERSION_SUB_LIST 2 CUTLASS_C4X_PATCH_VERSION)
target_compile_definitions(cutlass_c4x INTERFACE
    CUTLASS_MAJOR_VERSION=${CUTLASS_C4X_MAJOR_VERSION}
    CUTLASS_MINOR_VERSION=${CUTLASS_C4X_MINOR_VERSION}
    CUTLASS_PATCH_VERSION=${CUTLASS_C4X_PATCH_VERSION}
)

# setting deep gemm
if(GIT_FOUND)
  message(STATUS "IS_SUPPORT_DEEPGEMM ${IS_SUPPORT_DEEPGEMM}")

  if(IS_SUPPORT_DEEPGEMM STREQUAL "TRUE")
    message(STATUS "Running submodule update to fetch DeepGEMM")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive 3rdparty/DeepGEMM
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)

    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive 3rdparty/DeepGEMM failed with ${GIT_SUBMOD_RESULT}, please checkout DeepGEMM submodule")
    endif()

    execute_process(COMMAND python3 -m pip install --upgrade --force-reinstall --ignore-installed .
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/DeepGEMM
      RESULT_VARIABLE DEEPGEMM_INSTALL_RESULT)

    if(NOT DEEPGEMM_INSTALL_RESULT EQUAL "0")
      message(FATAL_ERROR "python -m pip install --upgrade --force-reinstall --ignore-installed -e . under DeepGEMM failed with ${DEEPGEMM_INSTALL_RESULT}, please checkout DeepGEMM install problem")
    endif()
    message(STATUS "Prepare DeepGEMM success")
  endif()
endif()

set(CUDA_INC_DIRS
  ${CUDA_PATH}/include
)

set(CUDA_LIB_DIRS
  ${CUDA_PATH}/lib64
)

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()
