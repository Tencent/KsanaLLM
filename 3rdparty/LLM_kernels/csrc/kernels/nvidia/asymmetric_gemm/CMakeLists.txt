#
# Copyright 2024 Tencent Inc.  All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# The Python executable will only be defined if building with Torch support. If
# not, we need to find it here.
if(NOT Python3_EXECUTABLE)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
endif()

# 获取所有相关目录下的所有文件
file(GLOB_RECURSE ALL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/*
  ${3RDPARTY_DIR}/cutlass/python/*
  ${CMAKE_CURRENT_SOURCE_DIR}/python/*
)

set(INSTANTIATION_GENERATION_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/cutlass_instantiations)
set(MD5_FILE ${INSTANTIATION_GENERATION_DIR}/.md5sums)

# 计算所有文件的组合MD5
function(calculate_files_md5 files_list output_md5)
  set(combined_content "")
  foreach(file ${files_list})
    if(EXISTS ${file})
      file(READ ${file} content)
      string(APPEND combined_content ${content})
    endif()
  endforeach()
  string(MD5 md5sum "${combined_content}")
  set(${output_md5} ${md5sum} PARENT_SCOPE)
endfunction()

# 检查是否需要重新生成代码
set(NEED_REGENERATE TRUE)
if(EXISTS ${MD5_FILE})
  # 读取之前保存的MD5值
  file(READ ${MD5_FILE} PREV_MD5)
  
  # 计算当前文件的MD5值
  calculate_files_md5("${ALL_FILES}" CURRENT_MD5)
  
  if(${PREV_MD5} STREQUAL ${CURRENT_MD5})
    set(NEED_REGENERATE FALSE)
    message(STATUS "Files MD5 unchanged, skipping regeneration")
  else()
    message(STATUS "Files MD5 changed, regenerating")
    message(STATUS "Previous MD5: ${PREV_MD5}")
    message(STATUS "Current MD5: ${CURRENT_MD5}")
  endif()
endif()

if(NEED_REGENERATE)
  message(STATUS "Regenerating CUTLASS kernels...")
  
  # 设置CUTLASS库
  execute_process(
    WORKING_DIRECTORY ${3RDPARTY_DIR}/cutlass/python/
    COMMAND ${Python3_EXECUTABLE} setup_library.py develop --user
    RESULT_VARIABLE _CUTLASS_LIBRARY_SUCCESS)

  if(NOT _CUTLASS_LIBRARY_SUCCESS MATCHES 0)
    message(
      FATAL_ERROR
        "Failed to set up the CUTLASS library due to ${_CUTLASS_LIBRARY_SUCCESS}."
    )
  endif()

  # 生成内核代码
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
    COMMAND ${Python3_EXECUTABLE} generate_kernels.py -o
            ${INSTANTIATION_GENERATION_DIR}
    RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

  if(NOT _KERNEL_GEN_SUCCESS MATCHES 0)
    message(
      FATAL_ERROR
        "Failed to generate CUTLASS kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
    )
  endif()

  # 更新MD5值
  calculate_files_md5("${ALL_FILES}" NEW_MD5)
  file(WRITE ${MD5_FILE} ${NEW_MD5})
else()
  message(STATUS "CUTLASS kernels are up to date, skipping generation")
endif()


# Get the sources for Mixed Input GEMM launchers
file(GLOB_RECURSE MIXED_CU_INSTANTIATIONS
     ${INSTANTIATION_GENERATION_DIR}/gemm/*.cu)
file(GLOB_RECURSE MIXED_SRC_CPP fpA_intB_gemm/*.cpp)
file(GLOB_RECURSE MIXED_SRC_CU fpA_intB_gemm/*.cu)

# Get the sources for MOE Grouped GEMM launchers
file(GLOB_RECURSE GROUPED_CU_INSTANTIATIONS
     ${INSTANTIATION_GENERATION_DIR}/gemm_grouped/*.cu)
file(GLOB_RECURSE GROUPED_SRC_CPP moe_gemm/*.cpp)
file(GLOB_RECURSE GROUPED_SRC_CU moe_gemm/*.cu)

# Get the sources for FP8 Rowwise GEMM launchers
file(GLOB_RECURSE FBGEMM_CU_INSTANTIATIONS
     ${INSTANTIATION_GENERATION_DIR}/fp8_rowwise_gemm/*.cu)
file(GLOB_RECURSE FBGEMM_SRC_CU fp8_rowwise_gemm/*.cu)

# Get the sources for all the remaining sources
file(GLOB_RECURSE SRC_CPP *.cpp)
file(GLOB_RECURSE SRC_CU *.cu)
set(ALL_SRCS ${SRC_CPP};${SRC_CU})
list(FILTER ALL_SRCS EXCLUDE REGEX "fpA_intB_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "moe_gemm/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX "fp8_rowwise_gemm/.*")
list(REMOVE_ITEM ALL_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/fused_gated_gemm/gemm_swiglu_e4m3.cu")

list(FILTER ALL_SRCS EXCLUDE REGEX ".*test.cu")

message(
  VERBOSE
  "Mixed srcs ${MIXED_SRC_CPP} ${MIXED_SRC_CU} ${MIXED_CU_INSTANTIATIONS}")
message(
  VERBOSE
  "Group srcs ${GROUPED_SRC_CU} ${GROUPED_SRC_CPP} ${GROUPED_CU_INSTANTIATIONS}"
)
message(VERBOSE "Fbgemm srcs ${FBGEMM_SRC_CU} ${FBGEMM_CU_INSTANTIATIONS}")
message(VERBOSE "All srcs ${ALL_SRCS}")

add_library(llm_kernels_nvidia_kernel_asymmetric_gemm STATIC ${ALL_SRCS} ${GROUPED_CU_INSTANTIATIONS} 
                                                             ${MIXED_CU_INSTANTIATIONS})

add_library(fpA_intB_gemm_src STATIC ${MIXED_SRC_CPP} ${MIXED_SRC_CU}
                                     ${MIXED_CU_INSTANTIATIONS})
# WARNING: Building with `-G` flag may generate invalid results for this target
add_library(moe_gemm_src STATIC ${GROUPED_SRC_CU} ${GROUPED_SRC_CPP}
                                ${GROUPED_CU_INSTANTIATIONS})
add_library(fb_gemm_src STATIC ${FBGEMM_SRC_CU} ${FBGEMM_CU_INSTANTIATIONS})

set(GEMM_SWIGLU_SM90_SRC_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_gated_gemm/gemm_swiglu_e4m3.cu)
add_library(gemm_swiglu_sm90_src STATIC ${GEMM_SWIGLU_SM90_SRC_CU})
foreach(target_name
        fpA_intB_gemm_src;moe_gemm_src;fb_gemm_src;gemm_swiglu_sm90_src;llm_kernels_nvidia_kernel_asymmetric_gemm)
  set_property(TARGET ${target_name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # Note - we deliberately do not include 90a PTX (even when 9.0+PTX is
  # specified). This is because sm_90a has arch conditional instructions that
  # are not forward compatible. As a result, it does not make sense to embed PTX
  # into the binary anyway.
  
  # support sm90(COMPILE_HOPPER_TMA_GEMMS)
  if("90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     OR "90-real" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG
     OR "90-real" IN_LIST CMAKE_CUDA_ARCHITECTURES_NATIVE)

    message(STATUS "MANUALLY APPENDING FLAG TO COMPILE FOR SM_90a.")
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_90a,code=sm_90a>)
    
    find_package(CUDAToolkit REQUIRED)

    # Hopper kernels require cuda lib for TMA APIs
    target_link_libraries(${target_name} PRIVATE cuda)

    # No kernels should be parsed, unless hopper is specified. This is a build
    # time improvement
    target_compile_definitions(${target_name} PRIVATE COMPILE_HOPPER_TMA_GEMMS)

    # define NDEBUG to skip cutlass brkpt when producer call consumer function
    # https://github.com/NVIDIA/cutlass/blob/v3.7.0/include/cutlass/pipeline/sm90_pipeline.hpp#L84
    target_compile_definitions(${target_name} PRIVATE NDEBUG)
  endif()
  

  # Suppress GCC note: the ABI for passing parameters with 64-byte alignment has
  # changed in GCC 4.6 This note appears for kernels using TMA and clutters the
  # compilation output.
  if(NOT WIN32)
    target_compile_options(
      ${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-psabi>)
  endif()

endforeach()

# target_link_libraries must be after set_property
target_link_libraries(llm_kernels_nvidia_kernel_asymmetric_gemm PUBLIC -lcublas -lcudart -lcublasLt
    llm_kernels_nvidia_utils)

# for test
file(GLOB_RECURSE ASYMMETRIC_GEMM_TEST_SRCS asymmetric_gemm_test.cu)
cc_test(llm_kernels_nvidia_kernel_asymmetric_gemm_test SRCS ${ASYMMETRIC_GEMM_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_asymmetric_gemm
    llm_kernels_nvidia_kernel_weight_only_batched_gemv
    llm_kernels_nvidia_kernel_permute
    llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_machete
    llm_kernels_nvidia_kernel_gemm_wrapper)

# for test
file(GLOB_RECURSE CUTLASS_PREPROCESSORS_TEST_SRCS cutlass_preprocessors_test.cu)
cc_test(llm_kernels_nvidia_kernel_cutlass_preprocessors_test SRCS ${CUTLASS_PREPROCESSORS_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_asymmetric_gemm)
