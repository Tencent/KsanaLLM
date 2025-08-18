/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

 #pragma once

 #include <cuda_runtime.h>
 
 namespace llm_kernels {
 namespace nvidia {

 // Gather submatrices from input tensor [M, N] according to data parallel (DP) group mapping.
 // Each DP group contributes a submatrix with its own sequence length.
 // Output shape: [m1 + m2 + ... + mk, N / tp_size]
 //   - m1, m2, ..., mk: sequence lengths from each DP group
 //   - N / tp_size: feature dimension after tensor parallel partitioning
 template <typename T>
 void InvokeGatherSubmatrix(const T* __restrict__ input, T* __restrict__ output, std::vector<size_t>& m_num_per_group,
                            size_t group_size, size_t group_num, size_t n_start, size_t n_end, size_t m, size_t n,
                            void* workspace, cudaStream_t& stream);

 /**
  * Redistributes input tensor by reorganizing prefill and decode tokens across DP groups.
  *
  * Transforms the interleaved token layout into a grouped layout where all prefill
  * tokens appear before all decode tokens.
  *
  * Input:  [mp0 + md0 + mp1 + md1+...+ mpK + mdK, N] - interleaved layout
  * Output: [mp0 + mp1 +...+ mpK + md0 + md1 + ... + mdK, N] - grouped layout
  *
  * Where:
  *   - mpk: prefill token count for k-th DP group
  *   - mdk: decode token count for k-th DP group
  *   - K: number of DP groups (0 to K)
  *   - N: feature dimension
  */
 template <typename T>
 void InvokeDpMapCopy(const T* __restrict__ input, T* __restrict__ output, const std::vector<size_t>& group_info,
                      size_t m, size_t n, void* workspace, cudaStream_t& stream);

 }  // namespace nvidia
 }  // namespace llm_kernels