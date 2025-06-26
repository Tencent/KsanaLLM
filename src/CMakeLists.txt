# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# import external dependencies
include(external/abseil-cpp)
include(external/protobuf)
include(external/tbb)
include(external/fmt)
include(external/loguru)
include(external/gflags)
include(external/httplib)
include(external/pybind11)
include(external/yaml-cpp)
include(external/nlohmann_json)
include(external/base64)
include(external/msgpack)
include(external/opentelemetry)
include(external/re2)
include(external/cppzmq)
include(external/libzmq)

# import subprojects
add_subdirectory(ksana_llm)
