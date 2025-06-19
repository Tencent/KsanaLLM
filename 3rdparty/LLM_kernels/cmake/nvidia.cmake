# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT WITH_CUDA)
  return()
endif()

find_package(CUDA 11.2 REQUIRED)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  # enable BFloat16
  add_definitions("-DENABLE_BF16")
  message(STATUS "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")

  # enable FP8
  if(${CUDA_VERSION_MINOR} VERSION_GREATER_EQUAL "8" OR ${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "12")
    add_definitions("-DENABLE_FP8")
    message(STATUS "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
  endif()
endif()

if(NOT DEFINED SM)
  execute_process(COMMAND python ${PROJECT_SOURCE_DIR}/tools/get_nvidia_gpu_properties.py OUTPUT_VARIABLE SM OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/triton_kernel_files)

# fetch 3rdparty
if(GIT_FOUND)
  message(STATUS "Running submodule update to fetch cutlass")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/cutlass
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)

  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

if(CUDA_PTX_VERBOSE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall --generate-line-info -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
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

# setting cutlass
set(CUTLASS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include)
set(CUTLASS_TOOLS_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include)
set(CUTLASS_EXTENSIONS_DIR ${PROJECT_SOURCE_DIR}/src/fastertransformer/cutlass_extensions/include)
subproject_version(${PROJECT_SOURCE_DIR}/3rdparty/cutlass CUTLASS_VERSION)
set(CUTLASS_VERSION_SUB_LIST ${CUTLASS_VERSION})
string(REPLACE "." ";" CUTLASS_VERSION_SUB_LIST "${CUTLASS_VERSION}")
message(STATUS "cutlass version is: ${CUTLASS_VERSION}")
list(GET CUTLASS_VERSION_SUB_LIST 0 CUTLASS_MAJOR_VERSION)
list(GET CUTLASS_VERSION_SUB_LIST 1 CUTLASS_MINOR_VERSION)
list(GET CUTLASS_VERSION_SUB_LIST 2 CUTLASS_PATCH_VERSION)
add_definitions("-DCUTLASS_MAJOR_VERSION=${CUTLASS_MAJOR_VERSION}")
add_definitions("-DCUTLASS_MINOR_VERSION=${CUTLASS_MINOR_VERSION}")
add_definitions("-DCUTLASS_PATCH_VERSION=${CUTLASS_PATCH_VERSION}")

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

    execute_process(COMMAND python setup.py develop
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/DeepGEMM
      RESULT_VARIABLE DEEPGEMM_INSTALL_RESULT)

    if(NOT DEEPGEMM_INSTALL_RESULT EQUAL "0")
      message(FATAL_ERROR "python setup.py develop under DeepGEMM failed with ${DEEPGEMM_INSTALL_RESULT}, please checkout DeepGEMM install problem")
    endif()

    execute_process(COMMAND pip install .
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/DeepGEMM
      RESULT_VARIABLE DEEPGEMM_INSTALL_RESULT)

    if(NOT DEEPGEMM_INSTALL_RESULT EQUAL "0")
      message(FATAL_ERROR "python setup.py install under DeepGEMM failed with ${DEEPGEMM_INSTALL_RESULT}, please checkout DeepGEMM install problem")
    endif()

    message(STATUS "Prepare DeepGEMM success")
  endif()
endif()

set(CUDA_INC_DIRS
  ${CUDA_PATH}/include
  ${CUTLASS_HEADER_DIR}
  ${CUTLASS_TOOLS_HEADER_DIR}
)

set(CUDA_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  # enable BFloat16
  add_definitions("-DENABLE_BFLOAT16")
  message(STATUS "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag")

  # enable FP8
  if(${CUDA_VERSION_MINOR} VERSION_GREATER_EQUAL "8" OR ${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "12")
    add_definitions("-DENABLE_FP8")
    message(STATUS "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
  endif()
endif()