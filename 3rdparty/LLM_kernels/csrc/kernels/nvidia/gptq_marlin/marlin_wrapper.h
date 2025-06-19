/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "csrc/utils/nvidia/scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin {

struct WorkspaceInfo {
  size_t c_tmp_size;
  size_t a_tmp_size;
  size_t workspace_size;
};

void awq_marlin_repack(const uint32_t* b_q_weight_ptr, uint32_t* out_ptr, int64_t size_k, int64_t size_n,
                       int64_t num_bits, int rank, cudaStream_t stream);

std::vector<int64_t> awq_marlin_repack_meta(int64_t size_k, int64_t size_n, int64_t num_bits);

void gptq_marlin_repack(const uint32_t* b_q_weight_ptr, const uint32_t* perm_ptr, uint32_t* out_ptr, int64_t size_k,
                        int64_t size_n, int64_t num_bits, bool has_perm, int rank, cudaStream_t stream);

std::vector<int64_t> gptq_marlin_repack_meta(int64_t size_k, int64_t size_n, int64_t num_bits);

template <typename T>
void gptq_marlin_gemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
                      bool has_zp, bool has_act_order, bool is_awq, int rank, cudaStream_t stream);

template <typename T>
WorkspaceInfo get_workspace(bool use_fp32_reduce, bool has_act_order, int rank, int64_t size_m, int64_t size_k);

template <typename T>
void permute_scales(cudaStream_t stream, const T* input, T* output, const size_t k, const size_t n,
                    const int64_t groupsize);

}  // namespace marlin
}  // namespace nvidia
}  // namespace llm_kernels
