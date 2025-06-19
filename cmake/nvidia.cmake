# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

if(CUDA_PTX_VERBOSE_INFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl -g3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD} -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3 -DCUDA_PTX_FP8_F2FP_ENABLED")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler --generate-line-info -Wall -DCUDA_PTX_FP8_F2FP_ENABLED")

# set CUDA related
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
set(SM_SETS 80 86 89 90 90a)
set(IS_SUPPORT_DEEPGEMM "FALSE")

# check if custom define SM
if(NOT DEFINED SM)
  foreach(SM_NUM IN LISTS SM_SETS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_NUM},code=sm_${SM_NUM}")

    if(SM_NUM VERSION_GREATER_EQUAL "90")
      set(IS_SUPPORT_DEEPGEMM "TRUE")
    endif()

    list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM_NUM})
    message(STATUS "Assign GPU architecture (sm=${SM_NUM})")
    string(REGEX MATCHALL "[0-9]" SUB_VER_NUM "${SM_NUM}")
    list(JOIN SUB_VER_NUM "." SM_ARCH_VER)
    list(APPEND TORCH_CUDA_ARCH_LIST ${SM_ARCH_VER})
  endforeach()
elseif("${SM}" MATCHES ",")
  # Multiple SM values
  string(REPLACE "," ";" SM_LIST ${SM})

  foreach(SM_NUM IN LISTS SM_LIST)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${SM_NUM},code=sm_${SM_NUM}")

    if(SM_NUM VERSION_GREATER_EQUAL "90")
      set(IS_SUPPORT_DEEPGEMM "TRUE")
    endif()

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
  string(REGEX MATCHALL "[0-9]" SUB_VER_NUM "${SM}")
  list(JOIN SUB_VER_NUM "." SM_ARCH_VER)
  list(APPEND TORCH_CUDA_ARCH_LIST ${SM_ARCH_VER})
endif()

set(CUDA_INC_DIRS
  ${CUDA_PATH}/include
  ${CUTLASS_HEADER_DIR}
)

set(CUDA_LIB_DIRS
  ${CUDA_PATH}/lib64
)

add_definitions("-DENABLE_CUDA")

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
