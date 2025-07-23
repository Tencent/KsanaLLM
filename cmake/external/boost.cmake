# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# 首先尝试使用系统已安装的 Boost，优先使用系统版本
find_package(Boost 1.75.0 COMPONENTS system filesystem dll QUIET)

if(Boost_FOUND)
    message(STATUS "使用系统已安装的 Boost ${Boost_VERSION}")
    message(STATUS "Boost include directory: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
    # 如果系统没有合适的 Boost 版本，则使用 FetchContent
    message(STATUS "系统未找到合适的 Boost 版本，使用 FetchContent 下载构建")

    include(FetchContent)

    set(BOOST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/boost)

    if(NOT DEFINED BOOST_VER)
        set(BOOST_VER 1.88.0)
    endif()

    # 根据平台选项选择不同的引入方式
    if(WITH_ACL)
        # ACL 平台使用 GIT 方式引入 Boost
        message(STATUS "使用 GIT 方式引入 Boost (ACL 平台)")

        # 将版本号转换为 Git 标签格式 (例如: 1.88.0 -> boost-1.88.0)
        string(REPLACE "." "_" BOOST_TAG_VER ${BOOST_VER})
        set(BOOST_GIT_TAG "boost-${BOOST_VER}")

        FetchContent_Declare(
            boost
            GIT_REPOSITORY https://github.com/boostorg/boost.git
            GIT_TAG ${BOOST_GIT_TAG}
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
            SOURCE_DIR ${BOOST_INSTALL_DIR}
        )
    else()
        # 其他平台（CUDA、TOPS 等）使用 URL 方式引入 Boost（更快，适合 CI/CD）
        if(WITH_CUDA)
            message(STATUS "使用 URL 方式引入 Boost (CUDA 平台)")
        elseif(WITH_TOPS)
            message(STATUS "使用 URL 方式引入 Boost (TOPS 平台)")
        else()
            message(STATUS "使用 URL 方式引入 Boost (默认平台)")
        endif()

        FetchContent_Declare(
            boost
            URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VER}/boost-${BOOST_VER}-cmake.tar.gz
            SOURCE_DIR ${BOOST_INSTALL_DIR}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
    endif()

    FetchContent_GetProperties(boost)
    if(NOT boost_POPULATED)
        if(WITH_ACL)
            message(STATUS "正在通过 GIT 克隆 Boost ${BOOST_VER}...")
        else()
            message(STATUS "正在通过 URL 下载 Boost ${BOOST_VER}...")
        endif()

        FetchContent_Populate(boost)

        # 配置构建选项
        set(BOOST_ENABLE_CMAKE ON)
        set(BUILD_SHARED_LIBS OFF)
        set(BOOST_INCLUDE_LIBRARIES system filesystem dll)

        add_subdirectory(${boost_SOURCE_DIR} ${boost_BINARY_DIR})

        message(STATUS "Boost source directory: ${boost_SOURCE_DIR}")
        message(STATUS "Boost binary directory: ${boost_BINARY_DIR}")

        # 添加头文件路径（为了兼容性）
        include_directories(${boost_SOURCE_DIR})
    endif()

    # 使用 add_subdirectory 构建的 Boost 不需要再次 find_package
    # 直接检查目标是否存在
    if(NOT TARGET Boost::dll)
        message(FATAL_ERROR "构建 Boost 后仍然无法找到 Boost::dll 目标")
    endif()
endif()

# 验证 Boost::dll 目标是否可用
if(NOT TARGET Boost::dll)
    message(FATAL_ERROR "Boost::dll 目标不可用，请检查 Boost 配置")
endif()

# 输出调试信息
message(STATUS "Boost::dll 目标已可用")
if(TARGET Boost::dll)
    get_target_property(BOOST_DLL_DEPS Boost::dll INTERFACE_LINK_LIBRARIES)
    message(STATUS "Boost::dll 依赖: ${BOOST_DLL_DEPS}")
endif()
