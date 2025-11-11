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

# enable FP8
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA version: ${CUDA_VERSION} is greater or equal than 11.8, enable -DENABLE_FP8 flag")
endif()


# DeepGEMM support - only for SM90/SM90a
set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG 0)
set(_deepgemm_candidate_arches ${CMAKE_CUDA_ARCHITECTURES})
if(DEFINED SM AND NOT SM STREQUAL "")
  string(REPLACE "," ";" _sm_arch_list "${SM}")
  list(APPEND _deepgemm_candidate_arches ${_sm_arch_list})
endif()

foreach(ARCH IN LISTS _deepgemm_candidate_arches)
  if(ARCH STREQUAL "90" OR ARCH STREQUAL "90a")
    set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG 1)
    break()
  endif()
endforeach()

if(ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
  message(STATUS "DeepGEMM support enabled (SM90/90a detected).")
else()
  message(STATUS "DeepGEMM support disabled (requires SM90/90a).")
endif()

add_compile_definitions(ENABLE_DEEPSEEK_DEEPGEMM=${ENABLE_DEEPSEEK_DEEPGEMM_FLAG})
set(ENABLE_DEEPSEEK_DEEPGEMM_FLAG ${ENABLE_DEEPSEEK_DEEPGEMM_FLAG} CACHE INTERNAL "Toggle DeepSeek DeepGEMM kernels" FORCE)
unset(_deepgemm_candidate_arches)

find_package(Git QUIET)
if(GIT_FOUND AND ENABLE_DEEPSEEK_DEEPGEMM_FLAG)
    set(DEEPGEMM_SOURCE_DIR ${CMAKE_BINARY_DIR}/third_party/DeepGEMM)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third_party)

    if(NOT EXISTS ${DEEPGEMM_SOURCE_DIR})
        message(STATUS "Cloning DeepGEMM repository...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --recursive https://github.com/deepseek-ai/DeepGEMM.git ${DEEPGEMM_SOURCE_DIR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone DeepGEMM repository.")
        endif()
    else()
        message(STATUS "DeepGEMM repository already exists, skipping clone.")
    endif()

    # Check if DeepGEMM is already built
    set(DEEPGEMM_BUILT_MARKER ${DEEPGEMM_SOURCE_DIR}/build)
    if(EXISTS ${DEEPGEMM_BUILT_MARKER})
        message(STATUS "DeepGEMM already built, skipping build process.")
    else()
        message(STATUS "Updating DeepGEMM submodules...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMODULE_RESULT
        )
        if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
            message(WARNING "Failed to update DeepGEMM submodules.")
        endif()
        message(STATUS "Checking out DeepGEMM to version v2.1.0...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} checkout v2.1.0
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE GIT_CHECKOUT_RESULT
        )
        if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
            message(WARNING "Failed to checkout DeepGEMM to v2.1.0, using current branch.")
        endif()

        message(STATUS "Applying DeepGEMM patch...")
        set(DEEPGEMM_PATCH_FILE ${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels/3rdparty/deepgemm/deepgemm.patch)
        if(EXISTS ${DEEPGEMM_PATCH_FILE})
            execute_process(
                COMMAND ${GIT_EXECUTABLE} apply ${DEEPGEMM_PATCH_FILE}
                WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
                RESULT_VARIABLE GIT_APPLY_RESULT
            )
            if(NOT GIT_APPLY_RESULT EQUAL 0)
                message(WARNING "Failed to apply DeepGEMM patch, continuing anyway.")
            else()
                message(STATUS "DeepGEMM patch applied successfully.")
            endif()
        else()
            message(WARNING "DeepGEMM patch file not found at ${DEEPGEMM_PATCH_FILE}, skipping patch.")
        endif()

        message(STATUS "Building and installing DeepGEMM...")
        execute_process(
            COMMAND bash develop.sh
            WORKING_DIRECTORY ${DEEPGEMM_SOURCE_DIR}
            RESULT_VARIABLE DEEPGEMM_INSTALL_RESULT
        )
        if(NOT DEEPGEMM_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build and install DeepGEMM.")
        endif()
        message(STATUS "DeepGEMM built and installed successfully.")
    endif()

    set(DEEPGEMM_SOURCE_DIR ${DEEPGEMM_SOURCE_DIR} CACHE PATH "DeepGEMM source directory" FORCE)
    set(DEEPGEMM_LIBRARY_ROOT_PATH ${DEEPGEMM_SOURCE_DIR}/deep_gemm CACHE PATH "DeepGEMM library root path" FORCE)
    set(DEEPGEMM_INCLUDE_DIRS
        ${DEEPGEMM_SOURCE_DIR}/csrc
        ${DEEPGEMM_SOURCE_DIR}/deep_gemm/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/cutlass/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/cutlass/tools/util/include
        ${DEEPGEMM_SOURCE_DIR}/third-party/fmt/include
        CACHE INTERNAL "DeepGEMM include directories")
endif()
