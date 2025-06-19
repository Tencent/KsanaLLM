/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cuda_runtime.h>

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
 * @param stream CUDA stream
 * @param eps Epsilon value to avoid division by zero
 * @param min_fp8 Minimum value for fp8
 * @param max_fp8 Maximum value for fp8
 */
template <typename T>
void per_token_group_quant_fp8(void* input, void* output_q, void* output_s, int m, int n, int64_t group_size,
                               bool is_column_major, cudaStream_t stream, float eps = 1e-10, float min_fp8 = -448.0,
                               float max_fp8 = 448.0);
}  // namespace nvidia
}  // namespace llm_kernels