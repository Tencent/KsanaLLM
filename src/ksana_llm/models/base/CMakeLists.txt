# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE model_base_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/*.cpp
)

list(FILTER model_base_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS, "")
set(kernels_ascend_LIBS, "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm)
endif()

if(WITH_ACL)
 list(APPEND kernels_ascend_LIBS atb_plugin_operations)
endif()

add_library(model_base STATIC ${model_base_SRCS})
target_link_libraries(model_base PUBLIC layers  ${kernels_nvidia_LIBS})


# for test
if(WITH_STANDALONE_TEST)
  set(MODELS_TEST_DEPS data_hub)
  set(MODEL_TEST_MAIN ${PROJECT_SOURCE_DIR}/tests/test.cpp)

  if(WITH_VLLM_FLASH_ATTN)
    set(MODELS_TEST_LIBS ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY})
  else()
    set(MODELS_TEST_LIBS "")
  endif()
  cpp_test(model_input_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/model_input_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    
endif()


