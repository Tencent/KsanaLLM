# Copyright 2024 Tencent Inc.  All rights reserved.

# set kernels target
file(GLOB_RECURSE FLASH_MLA_SRCS *.cu)
list(FILTER FLASH_MLA_SRCS EXCLUDE REGEX ".*test.cu")

# Create an empty library first
add_library(llm_kernels_nvidia_kernel_flash_mla STATIC)
set_property(TARGET llm_kernels_nvidia_kernel_flash_mla PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET llm_kernels_nvidia_kernel_flash_mla PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(llm_kernels_nvidia_kernel_flash_mla PROPERTIES LINKER_LANGUAGE CUDA)

# 根据SM变量选择合适的CUDA架构
if(NOT DEFINED SM)
  set(SM "90a")  # 默认使用90a
endif()

string(SUBSTRING "${SM}" 0 1 SM_FIRST_CHAR)
if(SM_FIRST_CHAR STREQUAL "9")
  set(CUDA_FLASH_MLA_FLAGS "-gencode arch=compute_90a,code=sm_90a -DENABLE_FLASH_MLA")
  add_definitions(-DENABLE_FLASH_MLA)
  message(STATUS "Flash MLA enabled for SM90+ GPUs")
else()
  set(CUDA_FLASH_MLA_FLAGS "-gencode arch=compute_86,code=sm_86")
  message(STATUS "Flash MLA disabled for SM${SM} GPUs (requires SM90+)")
endif()

# Define custom CUDA flags with absolute include paths
set(CUSTOM_CUDA_FLAGS "-I${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels/3rdparty/cutlass/include \
-I${CMAKE_SOURCE_DIR}/3rdparty/LLM_kernels \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options '-fPIC' \
-O3 -std=c++17 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations \
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
--ptxas-options=-v,--register-usage-level=10 \
${CUDA_FLASH_MLA_FLAGS} --threads 16")

# Convert the string to a list of arguments
separate_arguments(CUSTOM_CUDA_FLAGS_LIST UNIX_COMMAND "${CUSTOM_CUDA_FLAGS}")

# For each source file, create a custom command to compile it with our custom flags
foreach(src_file ${FLASH_MLA_SRCS})
  get_filename_component(src_name ${src_file} NAME_WE)
  set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${src_name}.o")
  
  # Custom command to compile each source file with our custom flags
  add_custom_command(
    OUTPUT ${obj_file}
    COMMAND ${CMAKE_CUDA_COMPILER} ${CUSTOM_CUDA_FLAGS_LIST} -c ${src_file} -o ${obj_file}
    DEPENDS ${src_file}
    COMMENT "Compiling ${src_file} with nvcc custom flags"
    VERBATIM
  )
  
  # Add the object file to the library
  target_sources(llm_kernels_nvidia_kernel_flash_mla PRIVATE ${obj_file})
endforeach()

target_link_libraries(llm_kernels_nvidia_kernel_flash_mla PUBLIC -lcublas -lcudart -lcublasLt)

# for test
file(GLOB_RECURSE FLASH_MLA_TEST_SRCS *test.cu)
cc_test(llm_kernels_nvidia_kernel_flash_mla_test SRCS ${FLASH_MLA_TEST_SRCS} DEPS
    llm_kernels_nvidia_utils llm_kernels_nvidia_kernel_flash_mla)
