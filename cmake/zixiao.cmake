# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

set(TOPS_PATH "/opt/tops")

set(TOPS_INC_DIRS
  ${TOPS_PATH}/include
)

set(TOPS_LIB_DIRS
  ${TOPS_PATH}/lib
)

include(FetchContent)
include(internal)

set(KLLM_INTERNAL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/ksanallm-internal)

FetchContent_Declare(
  kllm_internal
  GIT_REPOSITORY https://git.woa.com/RondaServing/LLM/KsanaLLM-internal.git
  GIT_TAG ${KSANA_INTERNAL_PACKAGE_COMMIT_ID}
  SOURCE_DIR ${KLLM_INTERNAL_INSTALL_DIR}
)

add_definitions("-DENABLE_TOPS")
add_definitions("-DENABLE_BFLOAT16")
