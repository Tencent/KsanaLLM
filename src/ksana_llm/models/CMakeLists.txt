# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE models_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/communicator/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/weight_method/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama4/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/baichuan/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/bge_reranker_minicpm/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/chatglm/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/gpt/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/mixtral/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen2_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen3_moe/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/internlm2/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/internlmxcomposer2/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/hunyuan_large/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/deepseek_v3/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/new_deepseek_v3/*.cpp
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/hunyuan_turbo/*.cpp)

list(FILTER models_SRCS EXCLUDE REGEX ".*test.cpp")

set(kernels_nvidia_LIBS "")
set(kernels_ascend_LIBS "")

if(WITH_CUDA)
  list(APPEND kernels_nvidia_LIBS llm_kernels_nvidia_kernel_asymmetric_gemm
                                  llm_kernels_nvidia_kernel_gemm_finegrained_mixed_dtype_gemm)
endif()

add_library(models STATIC ${models_SRCS})
target_link_libraries(models PUBLIC modules model_base cache_manager
  -Wl,--whole-archive
  layers
  -Wl,--no-whole-archive
  samplers data_hub ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${kernels_nvidia_LIBS})

# for test
if(WITH_STANDALONE_TEST)
  set(MODELS_TEST_DEPS models runtime data_hub)
  set(MODEL_TEST_MAIN ${PROJECT_SOURCE_DIR}/tests/test.cpp)

  if(WITH_CUDA)
    set(MODELS_TEST_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  else()
    set(MODELS_TEST_LIBS "")
  endif()

  # 调用 cpp_test
  cpp_test(fake_weight_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/base/fake_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  cpp_test(expert_para_weight_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common_moe/common_moe_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS}  LIBS ${MODELS_TEST_LIBS} data_hub)
  cpp_test(common_weight_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common/common_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  cpp_test(llama_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/llama/llama_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS} ${TORCH_LIBRARIES})
  cpp_test(simple_decoder_layer_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/common/simple_decoder_layer_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
  if(WITH_CUDA)
    cpp_test(hunyuan_turbo_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/hunyuan_turbo/hunyuan_turbo_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(qwen_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen/qwen_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    cpp_test(new_qwen_config_parser_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/qwen/new_qwen_config_parser_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

    if(DEFINED SM AND "${SM}" STREQUAL "90a")
      cpp_test(deepseek_v3_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/deepseek_v3/deepseek_v3_model_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(deepseek_v3_dp_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/deepseek_v3/deepseek_v3_dp_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
      cpp_test(quant_int4_weight_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/quant_int4_weight_test.cpp ${MODEL_TEST_MAIN} DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
    endif()
  endif()
endif()

if(WITH_CUDA)
  file(GLOB_RECURSE quant_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/quant_fp8_weight_test.cpp)
  message(STATUS "quant_test_SRCS: ${quant_test_SRCS} ")

  if(WITH_STANDALONE_TEST)
    cpp_test(cutlass_utils_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/cutlass_utils_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(marlin_utils_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/quant/marlin_utils_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(quant_fp8_weight_test SRCS ${quant_test_SRCS} DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    cpp_test(common_method_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/weight_method/common_method_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    if(${CUDA_VERSION} VERSION_GREATER_EQUAL "12.8")
      cpp_test(w4a8_awq_method_test SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/models/weight_method/w4a8_awq_method_test.cpp DEPS models runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    endif()
  endif()
endif()
