# 确保包含 FetchContent 模块
include(FetchContent)

set(OPENTELEMETRY_CPP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/opentelemetry-cpp)

# 设置变量，确保在 FetchContent_MakeAvailable 之前
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for opentelemetry-cpp" FORCE)
set(WITH_EXAMPLES OFF CACHE BOOL "Do not build examples" FORCE)
set(WITH_DEPRECATED_SDK_FACTORY OFF CACHE BOOL "Disable deprecated SDK factory" FORCE)
set(WITH_OTLP_HTTP ON CACHE BOOL "Build OTLP HTTP exporter" FORCE)

set(absl_FOUND TRUE)
add_compile_definitions(HAVE_ABSEIL)
set(OPENTELEMETRY_ABI_VERSION_NO 1)
set(WITH_ABSEIL ON CACHE BOOL "Enable absl" FORCE)

FetchContent_Declare(
  opentelemetry-cpp
  GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
  GIT_TAG v1.16.0
  GIT_SUBMODULES ""
  SOURCE_DIR ${OPENTELEMETRY_CPP_INSTALL_DIR}
)

FetchContent_MakeAvailable(opentelemetry-cpp)

message(STATUS "Opentelemetry-cpp source directory: ${opentelemetry-cpp_SOURCE_DIR}")
message(STATUS "Opentelemetry-cpp binary directory: ${opentelemetry-cpp_BINARY_DIR}")

# 设置 opentelemetry-cpp 的路径
set(OPENTELEMETRY_CPP_INCLUDE_DIRS
    ${opentelemetry-cpp_SOURCE_DIR}/api/include
    ${opentelemetry-cpp_SOURCE_DIR}/sdk/include
    ${opentelemetry-cpp_SOURCE_DIR}/exporters/ostream/include
    ${opentelemetry-cpp_SOURCE_DIR}/exporters/otlp/include
    ${opentelemetry-cpp_SOURCE_DIR}/ext/include
)
include_directories(${OPENTELEMETRY_CPP_INCLUDE_DIRS})

# 设置 opentelemetry-cpp 的所有库
set(OPENTELEMETRY_CPP_LIBRARIES
opentelemetry_exporter_ostream_metrics
opentelemetry_exporter_otlp_http
opentelemetry_exporter_otlp_http_metric
opentelemetry_exporter_otlp_http_client
opentelemetry_http_client_curl
opentelemetry_metrics
opentelemetry_otlp_recordable
opentelemetry_resources
opentelemetry_common
opentelemetry_version
)
