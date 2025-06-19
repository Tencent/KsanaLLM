# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# prepare cutlass
message(STATUS "Running submodule update to fetch FlashAttention")
find_package(Git QUIET)

set(FLASH_ATTN_PYTHON_SO, "")
set(FLASH_ATTN_VERSION, "")
set(FLASH_ATTN_MINOR_VERSION, "")

if(WITH_VLLM_FLASH_ATTN)
  # NOTE(karlluo): build with https://github.com/vllm-project/flash-attention/blob/v2.6.2/build.sh and change the PYTORCH_VERSION and MAIN_CUDA_VERSION for dedicate environment
  # for 2.7.2 build from 175ebb204693d0c28b4e367fa8cb6af8e4366e92 https://github.com/vllm-project/flash-attention
  # for 2.6.2 build from v2.6.2 https://github.com/vllm-project/flash-attention
  execute_process(COMMAND python -c "import torch,vllm_flash_attn_2_cuda;print(vllm_flash_attn_2_cuda.__file__)" OUTPUT_VARIABLE FLASH_ATTN_PYTHON_SO OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND python -c "import vllm_flash_attn;print(vllm_flash_attn.__version__)" OUTPUT_VARIABLE FLASH_ATTN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND python -c "import vllm_flash_attn;print(vllm_flash_attn.__version__.split('.')[1])" OUTPUT_VARIABLE FLASH_ATTN_MINOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(COMMAND python -c "import torch,flash_attn_2_cuda;print(flash_attn_2_cuda.__file__)" OUTPUT_VARIABLE FLASH_ATTN_PYTHON_SO OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__)" OUTPUT_VARIABLE FLASH_ATTN_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND python -c "import flash_attn;print(flash_attn.__version__.split('.')[1])" OUTPUT_VARIABLE FLASH_ATTN_MINOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "FLASH_ATTN_PYTHON_SO: ${FLASH_ATTN_PYTHON_SO}")
if("${FLASH_ATTN_PYTHON_SO}" STREQUAL "")
  message(FATAL_ERROR "FLASH_ATTN_PYTHON_SO is empty, please check your python environment WITH_VLLM_FLASH_ATTN=${WITH_VLLM_FLASH_ATTN}")
endif()

add_library(flash_attn_kernels UNKNOWN IMPORTED)
set_property(TARGET flash_attn_kernels PROPERTY IMPORTED_LOCATION "${FLASH_ATTN_PYTHON_SO}")

if(WITH_VLLM_FLASH_ATTN)
  add_definitions("-DENABLE_VLLM_FLASH_ATTN_2")
  set(ENABLE_VLLM_FLASH_ATTN_2 TRUE)
  add_definitions("-DENABLE_VLLM_FLASH_ATTN_MINOR_${FLASH_ATTN_MINOR_VERSION}")
  message(STATUS "using vllm flash attention ${FLASH_ATTN_VERSION} from python")
  add_definitions("-DENABLE_FLASH_ATTN_WITH_CACHE")
else()
  add_definitions("-DENABLE_FLASH_ATTN_2")
  set(ENABLE_FLASH_ATTN_2 TRUE)
  add_definitions("-DENABLE_FLASH_ATTN_MINOR_${FLASH_ATTN_MINOR_VERSION}")
  message(STATUS "using flash attention ${FLASH_ATTN_VERSION} from python")
endif()
