# Copyright 2024 Tencent Inc.  All rights reserved.

file(GLOB_RECURSE ALL_BEST_GEMM_ALGO_SEARCH_SRCS search_best_gemm_algo.cc)
add_executable(search_best_gemm_algo ${ALL_BEST_GEMM_ALGO_SEARCH_SRCS})
target_link_libraries(search_best_gemm_algo PUBLIC -lcudart -lcublasLt -lnvidia-ml llm_kernels_nvidia_kernel_gemm_wrapper gflags ksana_llm_yaml)
set_property(TARGET search_best_gemm_algo PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET search_best_gemm_algo PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
