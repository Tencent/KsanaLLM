/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result,
                        cudaStream_t& stream);

/**
 * @brief Compute the local maximum values and their corresponding indices for each batch within a rank
 *
 * @tparam T             Data type of the input elements
 * @param input          Pointer to input tensor of shape [batch_size, vocab_size_pad]
 * @param batch_size     Number of batches
 * @param vocab_size     Actual vocabulary size (without padding)
 * @param vocab_size_pad Padded vocabulary size (may be > vocab_size)
 * @param rank           Rank/index of the current partition in tensor parallelism
 * @param result         Float pointer to output tensor of shape [batch_size, 2],
 *                       where each element stores (max_value, max_index) for each batch
 * @param stream         CUDA stream for kernel execution
 */
template <typename T>
void InvokeLocalArgMaxReduce(const T* input, const int32_t batch_size, const int32_t vocab_size,
                             const int32_t vocab_size_pad, const int32_t rank, float* result, cudaStream_t stream);

/**
 * @brief Compute the final argmax index across all ranks by reducing local max/index pairs
 *
 * @param input        Float pointer to input tensor of shape [batch_size, 2 * tp_size], interleaving pairs of (max,
 * idx) from all ranks
 * @param batch_size   Number of batches
 * @param tp_size      Number of ranks (tensor parallel size)
 * @param result       Pointer to output tensor of shape [batch_size], storing the global argmax indices
 * @param stream       CUDA stream for kernel execution
 */
void InvokeWarpArgMaxReduce(const float* input, const int32_t batch_size, const int32_t tp_size, uint32_t* result,
                            cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
