# 确保包含 FetchContent 模块
include(FetchContent)

set(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/protobuf)

# 使用 FetchContent 下载和构建 Protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable testing for protobuf" FORCE)

if(NOT DEFINED PROTOBUF_GIT_TAG)
    set(PROTOBUF_GIT_TAG v3.15.8)
endif()
set(PROTOBUF_GIT_URL https://github.com/protocolbuffers/protobuf/archive/${PROTOBUF_GIT_TAG}.tar.gz)

FetchContent_Declare(
  protobuf
  URL        ${PROTOBUF_GIT_URL}
  SOURCE_DIR ${PROTOBUF_INSTALL_DIR}
)

FetchContent_GetProperties(protobuf)
FetchContent_GetProperties(com_google_protobuf)
if((NOT protobuf_POPULATED) AND (NOT com_google_protobuf_POPULATED))
    FetchContent_Populate(protobuf)

    # Mark com_google_protobuf as populated, to prevent it being imported twice.
    define_property(GLOBAL PROPERTY _FetchContent_com_google_protobuf_populated
        BRIEF_DOCS "Indicates whether protobuf has been impoerted"
        FULL_DOCS  "This is defined by fetch content for protobuf used by trpc-cpp"
    )
    set_property(GLOBAL PROPERTY _FetchContent_com_google_protobuf_populated TRUE)
    set(com_google_protobuf_SOURCE_DIR ${protobuf_SOURCE_DIR})

    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
endif()

# 打印 Protobuf 的构建目录
message(STATUS "Protobuf source directory: ${protobuf_SOURCE_DIR}")
message(STATUS "Protobuf binary directory: ${protobuf_BINARY_DIR}")

# 手动设置 Protobuf 的路径
set(Protobuf_INCLUDE_DIR ${protobuf_SOURCE_DIR}/src)
set(Protobuf_INCLUDE_DIRS ${protobuf_SOURCE_DIR}/src)
set(Protobuf_IMPORT_DIRS ${Protobuf_INCLUDE_DIR})
set(PROTOBUF_PROTOC_EXECUTABLE protobuf::protoc)

include_directories(${Protobuf_INCLUDE_DIR})

# 设置 Protobuf 的所有库
set(Protobuf_LIBRARIES protobuf)
