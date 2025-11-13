add_definitions(-DENABLE_FP4=ON)

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
  ${3RDPARTY_DIR}/c4x/cutlass/python/*
)

set(INSTANTIATION_GENERATION_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/cutlass_kernels_instantiations)
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
  message(STATUS "Regenerating cutlass kernels...")
  
  # NOTE(karlluo): setup cutlass python environment refer to: https://github.com/NVIDIA/cutlass/tree/main/python#installation
  execute_process(
    WORKING_DIRECTORY ${3RDPARTY_DIR}/c4x/cutlass
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
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
    COMMAND ${Python3_EXECUTABLE} generate_kernels.py -o 
            ${INSTANTIATION_GENERATION_DIR} -a 90
    RESULT_VARIABLE _KERNEL_GEN_SUCCESS)

  if(NOT _KERNEL_GEN_SUCCESS MATCHES 0)
    message(
      FATAL_ERROR
        "Failed to generate cutlass kernel instantiations due to ${_KERNEL_GEN_SUCCESS}."
    )
  endif()

  # 更新MD5值
  calculate_files_md5("${ALL_FILES}" NEW_MD5)
  file(WRITE ${MD5_FILE} ${NEW_MD5})
else()
  message(STATUS "cutlass kernels are up to date, skipping generation")
endif()

file(GLOB_RECURSE CUTLASS_KERNELS_SRC_CPP *.cpp *.cu ${INSTANTIATION_GENERATION_DIR}/*.cu)
list(FILTER CUTLASS_KERNELS_SRC_CPP EXCLUDE REGEX ".*test.cu")

add_library(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels STATIC 
    ${CUTLASS_KERNELS_SRC_CPP})
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC fmt -lcublas -lcudart -lcublasLt -lnvrtc -lcuda
    cutlass_c4x llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_others_trtllm_dev_common
    llm_kernels_nvidia_kernel_others_trtllm_dev_utils)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC COMPILE_HOPPER_TMA_GEMMS)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC COMPILE_HOPPER_TMA_GROUPED_GEMMS)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC NDEBUG)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC ENABLE_BF16)
target_compile_definitions(llm_kernels_nvidia_kernel_others_trtllm_dev_cutlass_kernels PUBLIC ENABLE_FP4)