if(WITH_INTERNAL_LIBRARIES)
    include(FetchContent)

    set(POLARIS_CPP_REPOSITORY_DIR ${polaris_cpp_SOURCE_DIR})
    set(POLARIS_CPP_WITH_EXAMPLES OFF)
    set(POLARIS_CPP_WITH_TESTS OFF)
    set(POLARIS_CPP_WITH_BENCHMARK OFF)
    set(ENABLE_STATIC_LIB ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        polaris_cpp
        # Use git@git.woa.com:polaris/polaris-cpp.git when clone repositories with ssh key
        GIT_REPOSITORY "https://git.woa.com/polaris/polaris-cpp.git"
        GIT_TAG "release_0.17.6"
    )

    if(NOT polaris_cpp_POPULATED)
        FetchContent_Populate(polaris_cpp)
    endif()

    # 根据 CMake 版本设置 C 标准
    if(CMAKE_VERSION VERSION_LESS "3.21.0")
        set(POLARIS_C_STANDARD "11")
        message(STATUS "CMake ${CMAKE_VERSION} < 3.21.0, using C11 standard for polaris-cpp")
    else()
        set(POLARIS_C_STANDARD "17")
        message(STATUS "CMake ${CMAKE_VERSION} >= 3.21.0, using C17 standard for polaris-cpp")
    endif()

    execute_process(
        COMMAND sed -i "s|set(POLARIS_CPP_MAX_C_STANDARD [0-9]\\+)|set(POLARIS_CPP_MAX_C_STANDARD ${POLARIS_C_STANDARD})|g" "${polaris_cpp_SOURCE_DIR}/CMakeLists.txt"
        COMMAND sed -i.bak "/^# Installation/,\$d" "${polaris_cpp_SOURCE_DIR}/CMakeLists.txt"
        COMMAND rm -rf "${polaris_cpp_SOURCE_DIR}/third_party/protobuf"
        COMMAND rm -rf "${polaris_cpp_SOURCE_DIR}/third_party/re2"
    )

    add_subdirectory("${polaris_cpp_SOURCE_DIR}" "${polaris_cpp_BINARY_DIR}")
endif()