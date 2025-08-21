# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE modules_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/modules/*.cpp
)

list(FILTER modules_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS, "")
set(kernels_ascend_LIBS, "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm)
endif()


if(WITH_ACL)
 list(APPEND kernels_ascend_LIBS atb_plugin_operations)
endif()

add_library(modules STATIC ${modules_SRCS})
target_link_libraries(modules PUBLIC layers ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS})

# for test
if(WITH_STANDALONE_TEST)
  set(MODELS_TEST_DEPS modules models runtime data_hub)
  set(MODEL_TEST_MAIN ${PROJECT_SOURCE_DIR}/tests/test.cpp)

  
  if(WITH_CUDA)
    set(MODELS_TEST_LIBS ${TORCH_LIBRARIES}  ${TORCH_PYTHON_LIBRARY})
  else()
    set(MODELS_TEST_LIBS  samplers ${kernels_ascend_LIBS})
  endif()
 if(WITH_CUDA)
  cpp_test(module_mla_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/modules/attention/multihead_latent_attention_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  cpp_test(module_bmm_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/modules/basic/bmm_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  endif()
endif()

