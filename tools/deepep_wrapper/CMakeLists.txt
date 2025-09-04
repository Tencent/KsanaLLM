# CMakeLists.txt - 避免conda cusparse版本

cmake_minimum_required(VERSION 3.18)

project(DeepEPWrapper 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Multi-process CUDA application with NVSHMEM"
)

# C++标准设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90")
endif()

# 手动设置NVSHMEM路径
set(NVSHMEM_ROOT "/usr/local/nvshmem")
message(STATUS "Using NVSHMEM from: ${NVSHMEM_ROOT}")

# 检查NVSHMEM是否存在
if(NOT EXISTS "${NVSHMEM_ROOT}")
    message(FATAL_ERROR "NVSHMEM not found at: ${NVSHMEM_ROOT}")
endif()

# 设置NVSHMEM路径
set(NVSHMEM_INCLUDE_DIR "${NVSHMEM_ROOT}/include")
set(NVSHMEM_LIBRARY_DIR "${NVSHMEM_ROOT}/lib")

# 检查关键文件
if(NOT EXISTS "${NVSHMEM_INCLUDE_DIR}/nvshmem.h")
    message(FATAL_ERROR "nvshmem.h not found at: ${NVSHMEM_INCLUDE_DIR}")
endif()

# 查找NVSHMEM库文件
find_library(NVSHMEM_LIBRARY
    NAMES nvshmem_host nvshmem
    PATHS ${NVSHMEM_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

if(NOT NVSHMEM_LIBRARY)
    message(FATAL_ERROR "NVSHMEM library not found in: ${NVSHMEM_LIBRARY_DIR}")
endif()

message(STATUS "NVSHMEM include: ${NVSHMEM_INCLUDE_DIR}")
message(STATUS "NVSHMEM library: ${NVSHMEM_LIBRARY}")

# 查找依赖
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找conda环境中的PyTorch和相关库
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FOUND
)

set(CUPTI_LIBRARY "")

if(TORCH_FOUND EQUAL 0)
    message(STATUS "Found PyTorch at: ${TORCH_PATH}")
    set(TORCH_LIB_PATH "${TORCH_PATH}/lib")
    
    # 查找conda环境中的CUPTI库
    set(CONDA_CUPTI_PATHS
        "${TORCH_PATH}/lib/../../nvidia/cuda_cupti/lib/libcupti.so.12"
        "${TORCH_PATH}/lib/../../nvidia/cuda_cupti/lib/libcupti.so"
        "${TORCH_PATH}/lib/libcupti.so.12"
        "${TORCH_PATH}/lib/libcupti.so"
    )
    
    foreach(cupti_path ${CONDA_CUPTI_PATHS})
        if(EXISTS "${cupti_path}")
            set(CUPTI_LIBRARY "${cupti_path}")
            message(STATUS "Found conda CUPTI: ${CUPTI_LIBRARY}")
            break()
        endif()
    endforeach()
    
    # 添加PyTorch库路径
    link_directories("${TORCH_LIB_PATH}")
    
    # 只添加安全的conda nvidia库路径（避免cusparse）
    set(SAFE_CONDA_NVIDIA_PATHS
        "${TORCH_PATH}/lib/../../nvidia/cuda_cupti/lib"
        "${TORCH_PATH}/lib/../../nvidia/cudnn/lib"
        "${TORCH_PATH}/lib/../../nvidia/cublas/lib"
        "${TORCH_PATH}/lib/../../nvidia/nccl/lib"
        # 不添加cusparse路径，使用系统版本
        "${TORCH_PATH}/lib/../../nvidia/cufft/lib"
        "${TORCH_PATH}/lib/../../nvidia/curand/lib"
    )
    
    foreach(nvidia_path ${SAFE_CONDA_NVIDIA_PATHS})
        if(EXISTS "${nvidia_path}")
            link_directories("${nvidia_path}")
            message(STATUS "Added conda lib path: ${nvidia_path}")
        endif()
    endforeach()
    
    # 确保系统CUDA库路径在前面
    link_directories(${CUDAToolkit_ROOT}/lib64)
    
else()
    message(WARNING "PyTorch not found, may cause issues with DeepEP")
endif()

if(NOT CUPTI_LIBRARY)
    # 回退到系统CUPTI
    find_library(CUPTI_LIBRARY
        NAMES cupti
        PATHS ${CUDAToolkit_ROOT}/extras/CUPTI/lib64
        NO_DEFAULT_PATH
    )
    if(CUPTI_LIBRARY)
        message(STATUS "Using system CUPTI: ${CUPTI_LIBRARY}")
    else()
        message(WARNING "CUPTI library not found")
    endif()
endif()

# DeepEP预编译库
set(DEEPEP_BASE_PATH "/usr/local/deepep/")
set(DEEPEP_LIB_PATH "${DEEPEP_BASE_PATH}/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so")
set(DEEPEP_INCLUDE_DIR "${DEEPEP_BASE_PATH}/csrc/" "${DEEPEP_BASE_PATH}/csrc/kernels/")
if(EXISTS "${DEEPEP_LIB_PATH}")
    message(STATUS "Found DeepEP library: ${DEEPEP_LIB_PATH}")
    add_library(DeepEP::deep_ep SHARED IMPORTED)
    set_target_properties(DeepEP::deep_ep PROPERTIES
        IMPORTED_LOCATION "${DEEPEP_LIB_PATH}"
    )
else()
    message(WARNING "DeepEP library not found: ${DEEPEP_LIB_PATH}")
endif()

# 包含目录
set(COMMON_INCLUDES
    ${NVSHMEM_INCLUDE_DIR}
    ${DEEPEP_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# 链接库
set(COMMON_LIBS
    CUDA::cudart
    CUDA::cuda_driver
    ${NVSHMEM_LIBRARY}
    Threads::Threads
    Python3::Python
    rt
    dl
)

# 添加CUPTI库
if(CUPTI_LIBRARY)
    list(APPEND COMMON_LIBS ${CUPTI_LIBRARY})
endif()

# 如果DeepEP库存在，添加到链接
if(TARGET DeepEP::deep_ep)
    list(APPEND COMMON_LIBS DeepEP::deep_ep)
endif()

# filesystem库支持
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    list(APPEND COMMON_LIBS stdc++fs)
endif()

# 库路径设置
link_directories(${NVSHMEM_LIBRARY_DIR})

# 添加子目录
add_subdirectory(src)

# 显示配置信息
message(STATUS "=== Configuration Summary ===")
message(STATUS "NVSHMEM Root: ${NVSHMEM_ROOT}")
message(STATUS "NVSHMEM Include: ${NVSHMEM_INCLUDE_DIR}")
message(STATUS "NVSHMEM Library: ${NVSHMEM_LIBRARY}")
message(STATUS "PyTorch Path: ${TORCH_PATH}")
message(STATUS "CUPTI Library: ${CUPTI_LIBRARY}")
message(STATUS "Using system CUDA libraries for cusparse/nvJitLink")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
