# src/CMakeLists.txt

# 设置Python路径
set(PYTHON_PATH "python" CACHE STRING "Python path")

# 获取PyTorch安装目录
execute_process(COMMAND ${PYTHON_PATH} "-c" 
    "import os; import torch; print(os.path.dirname(torch.__file__), end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Failed to get PyTorch directory.")
endif()

message(STATUS "PyTorch directory: ${TORCH_DIR}")

# 添加到CMAKE_PREFIX_PATH并查找PyTorch
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

# 收集源文件
set(PROJECT_SOURCES
    main.cpp
    process.cpp
    wrapper.cpp
    common.cpp
    deep_ep.cpp
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    ${COMMON_LIBS}
    ${TORCH_LIBRARIES}
)
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--allow-multiple-definition")

# 设置头文件路径
target_include_directories(${PROJECT_NAME} 
    PRIVATE
    ${COMMON_INCLUDES} 
    ${TORCH_INCLUDE_DIRS}
)

# 编译选项 - 去掉可能有问题的TORCH_CXX_FLAGS
target_compile_options(${PROJECT_NAME} 
    PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wno-unused-parameter
    >
)

# 设置C++标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 其余RPATH设置保持不变...
set(RPATH_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../lib")

if(TORCH_PATH)
    list(APPEND RPATH_DIRS "${TORCH_PATH}/lib")
    list(APPEND RPATH_DIRS "${TORCH_PATH}/lib/../../nvidia/cuda_cupti/lib")
    list(APPEND RPATH_DIRS "${TORCH_PATH}/lib/../../nvidia/cudnn/lib")
    list(APPEND RPATH_DIRS "${TORCH_PATH}/lib/../../nvidia/cublas/lib")
    list(APPEND RPATH_DIRS "${TORCH_PATH}/lib/../../nvidia/nccl/lib")
endif()

if(TORCH_DIR)
    list(APPEND RPATH_DIRS "${TORCH_DIR}/lib")
endif()

string(REPLACE ";" ":" RPATH_STRING "${RPATH_DIRS}")

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${RPATH_STRING}"
)

message(STATUS "Executable configured with PyTorch and conda support")
message(STATUS "PyTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
