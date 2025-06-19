# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)
include(ExternalProject)

set(YAML_INSTALL_DIR ${THIRD_PARTY_PATH}/download/yaml-cpp)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
option(YAML_CPP_BUILD_TESTS "Enable yaml testing" OFF)

# Keep the same yaml version with trpc
set(YAML_VER 0.7.0)
set(YAML_URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-${YAML_VER}.tar.gz)
FetchContent_Declare(
    yaml_cpp
    URL ${YAML_URL}
    SOURCE_DIR ${YAML_INSTALL_DIR}
)

FetchContent_MakeAvailable(yaml_cpp)
include_directories(${YAML_INSTALL_DIR}/include)

# the declare yaml-cpp will generate libyaml-cpp.a,
# the involked library name is yaml-cpp, we alias it as ksana_llm_yaml
add_library(ksana_llm_yaml ALIAS yaml-cpp)
