# Copyright 2025 Tencent Inc.  All rights reserved.

if (SM STREQUAL "90" OR SM STREQUAL "90a")
    add_definitions(-DENABLE_MACHETE=ON)
endif()

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
  ${3RDPARTY_DIR}/c3x/cutlass/python/*
)

set(INSTANTIATION_GENERATION_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/machete_instantiations)
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
  message(STATUS "Regenerating machete kernels...")
  
  # NOTE(karlluo): setup cutlass python environment refer to: https://github.com/NVIDIA/cutlass/tree/main/python#installation
  execute_process(
    WORKING_DIRECTORY ${3RDPARTY_DIR}/c3x/cutlass
    COMMAND ${Python3_EXECUTABLE} -m pip install -e .
    RESULT_VARIABLE _CUTLASS_LIBRARY_SUCCESS)

  if(NOT _CUTLASS_LIBRARY_SUCCESS MATCHES 0)
    message(
      FATAL_ERROR
        "Failed to set up the CUTLASS library due to ${_CUTLASS_LIBRARY_SUCCESS}."
    )
  endif()

  # 生成内核代码
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMAND ${Python3_EXECUTABLE} generate.py --output_dir
            ${INSTANTIATION_GENERATION_DIR}
    RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

  if(NOT _KERNEL_GEN_SUCCESS MATCHES 0)
    message(
      FATAL_ERROR
        "Failed to generate machete kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
    )
  endif()

  # 更新MD5值
  calculate_files_md5("${ALL_FILES}" NEW_MD5)
  file(WRITE ${MD5_FILE} ${NEW_MD5})
else()
  message(STATUS "machete kernels are up to date, skipping generation")
endif()

# set kernels target
file(GLOB_RECURSE MACHETE_SRCS *.cu)
file(GLOB_RECURSE MACHETE_INSTANTIATIONS ${INSTANTIATION_GENERATION_DIR}/*.cu)
list(FILTER MACHETE_SRCS EXCLUDE REGEX ".*test.cu")
add_library(llm_kernels_nvidia_kernel_machete STATIC ${MACHETE_SRCS} ${MACHETE_INSTANTIATIONS})
set_property(TARGET llm_kernels_nvidia_kernel_machete PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_machete PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_machete PUBLIC fmt -lcuda -lcudart -lnccl -lnvidia-ml 
    llm_kernels_nvidia_utils)
target_link_libraries(llm_kernels_nvidia_kernel_machete PRIVATE cutlass_c3x)

target_compile_options(llm_kernels_nvidia_kernel_machete PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# for test
file(GLOB_RECURSE ADD_TEST_SRCS machete_wrapper_test.cu)
cc_test(llm_kernels_nvidia_kernel_machete_test SRCS ${ADD_TEST_SRCS} DEPS 
    llm_kernels_nvidia_utils
    llm_kernels_nvidia_kernel_permute
    llm_kernels_nvidia_kernel_cast
    llm_kernels_nvidia_kernel_gemm_wrapper
    llm_kernels_nvidia_kernel_machete)