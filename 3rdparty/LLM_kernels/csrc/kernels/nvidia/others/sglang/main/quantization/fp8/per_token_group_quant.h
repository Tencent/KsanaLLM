/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Performs per-token-group quantization on input
 *     It converts the input data into signed float8 values and returns the quantized
 *     data in output_q along with the scaling factor in output_s used for quantization.
 * @tparam T Input data type (float, half, __nv_bfloat16)
 * @param input Input with shape [m, n]
 * @param output_q Quantized data with type fp8
 * @param output_s Scaling factor with type float
 * @param m Shape[0] of input
 * @param n Shape[1] of input
 * @param group_size Size of each group
 * @param is_column_major Whether the input is in column-major order
 * @param fuse_silu_mul Whether to apply silu-mul to the input fisrt
 * @param stream CUDA stream
 */
template <typename T>
void per_token_group_quant_fp8(const void* input, void* output_q, void* output_s, int m, int n, int64_t group_size,
                               bool is_column_major, bool fuse_silu_mul, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
