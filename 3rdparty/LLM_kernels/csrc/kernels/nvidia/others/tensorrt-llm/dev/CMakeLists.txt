# Copyright 2025 Tencent Inc.  All rights reserved.

add_subdirectory(common)
add_subdirectory(utils)
if (SM STREQUAL "90" OR SM STREQUAL "90a")
    add_subdirectory(cutlass_kernels)
    add_subdirectory(deep_gemm)
endif()