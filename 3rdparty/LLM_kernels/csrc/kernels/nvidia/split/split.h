/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

 #pragma once

 #include <cuda_runtime.h>
 
 namespace llm_kernels {
 namespace nvidia {

 // Split a matrix into multiple matrices along the column dimension
 template <typename T>
 void InvokeSplit(const T* __restrict__ input, const std::vector<T*>& output_ptrs,
                  std::vector<int>& col_offsets,  // [0, col1, col1+col2, ...]
                  int rows, int cols, int num_outputs, cudaStream_t& stream);
 }  // namespace nvidia
 }  // namespace llm_kernels