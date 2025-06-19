/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdint.h>
#include <cstddef>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

enum class QuantType { W8_A16, W4_A16, W4_AFP8 };

constexpr int get_weight_quant_bits(QuantType quant_type) {
  switch (quant_type) {
    case QuantType::W8_A16:
      return 8;
    case QuantType::W4_A16:
      return 4;
    case QuantType::W4_AFP8:
      return 4;
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false, "Invalid quant_type");
      return -1;
  }
}

struct LayoutDetails {
  enum class Layout { UNKNOWN, ROW_MAJOR, COLUMN_MAJOR };

  Layout layoutB = Layout::UNKNOWN;
  int rows_per_column_tile = 1;
  int columns_interleaved = 1;

  bool uses_imma_ldsm = false;
};

std::vector<int> get_permutation_map(QuantType quant_type);

LayoutDetails getLayoutDetailsForTransform(QuantType quant_type, int arch);

void fast_permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor, const int32_t* row_permutation, int row_permutation_size, 
  std::vector<size_t> const& shape, QuantType quant_type, int64_t const arch_version, cudaStream_t stream);

void fast_subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
                       std::vector<size_t> const& shape, QuantType quant_type, cudaStream_t stream);

void fast_interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor,
            std::vector<size_t> const& shape, QuantType quant_type, LayoutDetails details, cudaStream_t stream);

void fast_add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type, cudaStream_t stream);

void fast_preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t* row_major_quantized_weight, const int32_t* device_row_permutation, const int row_permutation_size,
  std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave, cudaStream_t stream);

// Shapes here can be 2 or 3D. 2-D shapes are [num_rows, num_cols]
// 3-D shapes are [num_experts, num_rows, num_cols]
void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, int8_t const* quantized_tensor,
                                   std::vector<size_t> const& shape, QuantType quant_type, const int64_t arch_version);

void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
                       std::vector<size_t> const& shape, QuantType quant_type);

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type);

void interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor,
  std::vector<size_t> const& shape, QuantType quant_type, LayoutDetails details);

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t const* row_major_quantized_weight,
                                       std::vector<size_t> const& shape, QuantType quant_type,
                                       bool force_interleave = false);

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, WeightType const* input_weight_ptr,
                        std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave);

// This is exposed so that we can write tests that use the processed weights for CUTLASS but the unprocessed weight
// to implement a simple reference implementation.
template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight,
                        ComputeType* scale_ptr, WeightType const* input_weight_ptr, std::vector<size_t> const& shape,
                        QuantType quant_type, bool force_interleave);

}  // namespace nvidia
}  // namespace llm_kernels
