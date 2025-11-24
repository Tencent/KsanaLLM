/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

/*
 * ******************************************************************* *
 * Fused ReduceScatter plus Fused(Residual, RMSNorm) plus AllGather    *
 * ******************************************************************* *
 *
 * 1. reduce_sactter(mcptr)
 * 2. residual += mcptr
 * 3. mcptr = norm(residual, weight, epsilon)
 * 4. gather(mcptr)
 */
template <typename T>
void FusedRsLmAgCta(int64_t mcptr,       // [token_num, hidden_size] multimem_ptr
                    void* residual,      // [token_num, hidden_size]
                    const void* weight,  // [hidden_size]
                    void* signal_pads,   // [token_num, hidden_size]
                    int rank, int world_size, float epsilon, int token_num, int hidden_size, cudaStream_t stream);

/*
 * ******************************************************************* *
 * Fused ReduceScatter plus Fused(Residual) plus AllGather             *
 * ******************************************************************* *
 *
 * 1. reduce_sactter(mcptr)
 * 2. mcptr += residual
 * 3. gather(mcptr)
 */
template <typename T>
void FusedRsAgCta(int64_t mcptr,      // [token_num, hidden_size] multimem_ptr
                  void* residual,     // [token_num, hidden_size]
                  void* signal_pads,  // [token_num, hidden_size]
                  int rank, int world_size, int token_num, int hidden_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
