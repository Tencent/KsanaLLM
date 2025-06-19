/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

#include "csrc/kernels/nvidia/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <int THREADS>
__global__ void permute_B_rows_for_mixed_gemm_kernel(
  const uint32_t* input_byte_ptr, uint32_t* output_byte_ptr,
  const int32_t* row_permutation,
  size_t num_experts, size_t num_rows, int num_vec_cols, int B_ROWS_PER_MMA) {

  int expert = blockIdx.x;
  int base_row = blockIdx.y * B_ROWS_PER_MMA;

  for (int write_col = threadIdx.x; write_col < num_vec_cols; write_col += THREADS) {
    for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row) {
      int write_row = base_row + tile_row;
      int tile_read_row = row_permutation[tile_row];
      int read_row = base_row + tile_read_row;
      int read_col = write_col;
  
      int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
      int64_t read_offset = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
      int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;
  
      output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
    }
  }
}

// shape是(num_experts, num_rows, num_cols)
// permuted_quantized_tensor 和 quantized_tensor 都是：[num_experts, num_rows, num_cols / pack_factor]
void fast_permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor, const int32_t* row_permutation, int row_permutation_size, 
                                   std::vector<size_t> const& shape, QuantType quant_type, int64_t const arch_version, cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
  int const K = 16 / BITS_PER_ELT;

  uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
  uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

  int MMA_SHAPE_N = 8;
  int B_ROWS_PER_MMA = 8 * K;
  int const elts_in_int32 = 32 / BITS_PER_ELT;

  int const num_vec_cols = num_cols / elts_in_int32;

  KLLM_KERNEL_CHECK_WITH_INFO(arch_version >= 75,
                       "Unsupported Arch. Pre-volta not supported. Column interleave not needed on Volta.");

  KLLM_KERNEL_CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0,
      fmtstr("Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of %d",
             B_ROWS_PER_MMA));
  KLLM_KERNEL_CHECK_WITH_INFO(num_cols % MMA_SHAPE_N == 0,
      fmtstr("Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of %d.",
             MMA_SHAPE_N));

  KLLM_KERNEL_CHECK_WITH_INFO(size_t(B_ROWS_PER_MMA) == row_permutation_size, "Unexpected number of LDSM rows permuted.");

  constexpr int THREADS = 64;
  dim3 gridSize(num_experts, num_rows / B_ROWS_PER_MMA);
  dim3 blockSize(THREADS);
  permute_B_rows_for_mixed_gemm_kernel<THREADS><<<gridSize, blockSize, 0, stream>>>(
    input_byte_ptr, output_byte_ptr, row_permutation,
    num_experts, num_rows, num_vec_cols, B_ROWS_PER_MMA);
}

template <typename T>
__device__ void swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

template <int THREADS, int bits_per_elt, int M_TILE_L1, int N_TILE_L1, int VECTOR_WIDTH, int ELTS_PER_BYTE>
__global__ void subbyte_transpose_impl_kernel(
  const uint8_t* input_byte_ptr, uint8_t* output_byte_ptr,
  size_t num_rows, size_t num_cols, size_t col_bytes, size_t col_bytes_trans) {

  const size_t expert = blockIdx.x;
  const size_t row_tile_start = blockIdx.y * M_TILE_L1;
  const size_t matrix_offset = expert * num_rows * col_bytes;

  uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

  int const row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);

  const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;
  int const col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

  if (row_tile_start < num_rows) {
    for (size_t col_tile_start_byte = threadIdx.x * N_TILE_L1; col_tile_start_byte < col_bytes; col_tile_start_byte += THREADS * N_TILE_L1) {
      int const col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        int const row = row_tile_start + ii;

        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          int const col = col_tile_start_byte + jj;

          const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

          if (row < row_limit && col < col_limit) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
            }
          }
        }
      }

      if constexpr (bits_per_elt == 8) {
        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          for (int jj = ii + 1; jj < N_TILE_L1; ++jj) {
            swap(cache_buf[ii][jj], cache_buf[jj][ii]);
          }
        }
      } else if constexpr (bits_per_elt == 4) {
        for (int ii = 0; ii < M_TILE_L1; ++ii) {
          // Using M_TILE_L1 here is deliberate since we assume that the cache tile
          // is square in the number of elements (not necessarily the number of bytes).
          for (int jj = ii + 1; jj < M_TILE_L1; ++jj) {
            int const ii_byte = ii / ELTS_PER_BYTE;
            int const ii_bit_offset = ii % ELTS_PER_BYTE;

            int const jj_byte = jj / ELTS_PER_BYTE;
            int const jj_bit_offset = jj % ELTS_PER_BYTE;

            uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
            uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
          }
        }
      }

      const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
      int const row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);

      for (int ii = 0; ii < M_TILE_L1; ++ii) {
        int const row = row_tile_start_trans + ii;
        for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH) {
          int const col = col_tile_start_byte_trans + jj;

          const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

          if (row < row_limit_trans && col < col_limit_trans) {
            for (int v = 0; v < VECTOR_WIDTH; ++v) {
              output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
            }
          }
        }
      }
    }
  }
}

template <QuantType quant_type>
void fast_subbyte_transpose_impl(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
                            std::vector<size_t> const& shape, cudaStream_t stream) {
  constexpr int bits_per_elt = get_weight_quant_bits(quant_type);

  KLLM_KERNEL_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  const size_t col_bytes = num_cols * bits_per_elt / 8;
  const size_t col_bytes_trans = num_rows * bits_per_elt / 8;

  uint8_t const* input_byte_ptr = reinterpret_cast<uint8_t const*>(quantized_tensor);
  uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

  static constexpr int ELTS_PER_BYTE = 8 / bits_per_elt;

  static constexpr int M_TILE_L1 = 64;
  static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;

  static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

  // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
  // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
  // allows GCC to emit vector instructions.
  KLLM_KERNEL_CHECK_WITH_INFO(
      !(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH),
      fmtstr("Number of bytes for rows and cols must be a multiple of %d. However, num_rows_bytes = %ld and "
             "num_col_bytes = %ld.",
             VECTOR_WIDTH, col_bytes_trans, col_bytes));

  int const num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
  int const num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

  constexpr int THREADS = 64;
  dim3 gridSize(num_experts, num_m_tiles);
  dim3 blockSize(THREADS);
  subbyte_transpose_impl_kernel<THREADS, bits_per_elt, M_TILE_L1, N_TILE_L1, VECTOR_WIDTH, ELTS_PER_BYTE><<<gridSize, blockSize, 0, stream>>>(input_byte_ptr, output_byte_ptr,
    num_rows, num_cols, col_bytes, col_bytes_trans);
}

// shape是(num_experts, num_rows, num_cols)
// transposed_quantized_tensor 和 quantized_tensor 都是：[num_experts, num_rows, num_cols / pack_factor]
void fast_subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
                       std::vector<size_t> const& shape, QuantType quant_type, cudaStream_t stream) {
  if (quant_type == QuantType::W8_A16) {
    fast_subbyte_transpose_impl<QuantType::W8_A16>(transposed_quantized_tensor, quantized_tensor, shape, stream);
  } else if (quant_type == QuantType::W4_A16) {
    fast_subbyte_transpose_impl<QuantType::W4_A16>(transposed_quantized_tensor, quantized_tensor, shape, stream);
  } else if (quant_type == QuantType::W4_AFP8) {
    fast_subbyte_transpose_impl<QuantType::W4_AFP8>(transposed_quantized_tensor, quantized_tensor, shape, stream);
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Invalid quant_type");
  }
}

template <int THREADS>
__global__ void interleave_column_major_tensor_kernel(
  const uint32_t* input_byte_ptr, uint32_t* output_byte_ptr,
  const int vec_rows_per_tile, const int num_vec_rows, const size_t num_cols, const int interleave) {

  int expert = blockIdx.x;
  int base_vec_row = blockIdx.y * vec_rows_per_tile;
  int vec_read_row = blockIdx.z + base_vec_row;

  if (vec_read_row < num_vec_rows) {
    for (int read_col = threadIdx.x; read_col < num_cols; read_col += THREADS) {
      const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
      const int64_t write_col = read_col / interleave;
      const int64_t vec_write_row = interleave * base_vec_row + vec_rows_per_tile * (read_col % interleave) + vec_read_row % vec_rows_per_tile;
      const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
      const int64_t write_offset = matrix_offset + int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
      output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
    }
  }
}

// shape是(num_experts, num_rows, num_cols)
// interleaved_quantized_tensor 和 quantized_tensor 都是：[num_experts, num_rows, num_cols / pack_factor]
void fast_interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor,
                                    std::vector<size_t> const& shape, QuantType quant_type, LayoutDetails details, cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
  const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
  const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
  const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

  int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
  int const elts_in_int32 = 32 / BITS_PER_ELT;

  int const rows_per_tile = details.rows_per_column_tile;

  KLLM_KERNEL_CHECK_WITH_INFO(!(num_rows % elts_in_int32),
      fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", elts_in_int32, num_rows));

  uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
  uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

  KLLM_KERNEL_CHECK_WITH_INFO(!(num_rows % rows_per_tile),
      fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", rows_per_tile, num_rows));

  int const num_vec_rows = num_rows / elts_in_int32;
  int const vec_rows_per_tile = rows_per_tile / elts_in_int32;
  int const interleave = details.columns_interleaved;

  constexpr int THREADS = 128;
  dim3 gridSize(num_experts, num_vec_rows / vec_rows_per_tile, vec_rows_per_tile);
  dim3 blockSize(THREADS);
  interleave_column_major_tensor_kernel<THREADS><<<gridSize, blockSize, 0, stream>>>(
    input_byte_ptr, output_byte_ptr,
    vec_rows_per_tile, num_vec_rows, num_cols, interleave);
}

__global__ void add_bias_and_interleave_int4s_inplace_step1_kernel(int8_t* inout_byte_ptr, const int num_bytes) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < num_bytes) {
    int8_t transformed_packed_int4s = 0;
    int8_t transformed_first_elt = (int8_t(inout_byte_ptr[ii] << 4) >> 4) + 8;
    int8_t transformed_second_elt = (inout_byte_ptr[ii] >> 4) + 8;

    transformed_packed_int4s |= transformed_first_elt;
    transformed_packed_int4s |= (transformed_second_elt << 4);
    inout_byte_ptr[ii] = transformed_packed_int4s;
  }
}

__global__ void add_bias_and_interleave_int4s_inplace_step2_kernel(uint32_t* register_ptr, const size_t num_registers) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < num_registers) {
    const uint32_t current_register = register_ptr[ii];
    uint32_t transformed_register = 0;

    for (int dest_idx = 0; dest_idx < 8; ++dest_idx) {
      int const src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
      int const src_shift = 4 * src_idx;
      int const dest_shift = 4 * dest_idx;

      const uint32_t src_bits = (current_register >> src_shift) & 0xF;
      transformed_register |= (src_bits << dest_shift);
    }
    register_ptr[ii] = transformed_register;
  }
}

// num_elts = num_experts * num_rows * num_cols
// packed_int4_tensor 是：[num_elts / pack_factor]
// packed_int4_tensor会被原地修改
void fast_add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts, cudaStream_t stream) {
  int const num_bytes = num_elts / 2;

  {
    constexpr int THREADS = 1024;
    dim3 gridSize((num_bytes + THREADS - 1) / THREADS);
    dim3 blockSize(THREADS);
    add_bias_and_interleave_int4s_inplace_step1_kernel<<<gridSize,blockSize,0,stream>>>(packed_int4_tensor,num_bytes);
  }

  KLLM_KERNEL_CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a multiple of 8 for register relayout");
  const size_t num_registers = num_bytes / 4;

  {
    constexpr int THREADS = 1024;
    dim3 gridSize((num_registers + THREADS - 1) / THREADS);
    dim3 blockSize(THREADS);
    add_bias_and_interleave_int4s_inplace_step2_kernel<<<gridSize,blockSize,0,stream>>>(reinterpret_cast<uint32_t*>(packed_int4_tensor),num_registers);
  }
}

__global__ void fast_swap_kernel(int8_t* input_byte_ptr, int8_t* output_byte_ptr, const size_t num_bytes) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < num_bytes) {
    swap(input_byte_ptr[ii], output_byte_ptr[ii]);
  }
}

void fast_swap(int8_t* src, int8_t* dst, const size_t num_bytes, cudaStream_t stream) {
  constexpr int THREADS = 1024;
  dim3 gridSize((num_bytes + THREADS - 1) / THREADS);
  dim3 blockSize(THREADS);
  fast_swap_kernel<<<gridSize,blockSize,0,stream>>>(src,dst,num_bytes);
}

void fast_add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type, cudaStream_t stream) {
  // TODO:W8情况没实现，有需要再做
  if (quant_type == QuantType::W4_A16) {
    fast_add_bias_and_interleave_int4s_inplace(tensor, num_elts, stream);
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
  }
}

// shape是(num_experts, num_rows, num_cols)
// force_interleave默认为false
// preprocessed_quantized_weight和row_major_quantized_weight的shape都是：[num_experts, num_rows, num_cols / pack_factor]
// device_row_permutation和row_permutation_size需要使用get_permutation_map函数获取并转换到cpu
// 输入权重row_major_quantized_weight会被当作workspace使用，导致其原数据不再可用
void fast_preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t* row_major_quantized_weight, const int32_t* device_row_permutation, const int row_permutation_size,
                                       std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave, cudaStream_t stream) {
  int arch = GetSMVersion();
  if (force_interleave && arch == 90) {
    // Workaround for MOE which doesn't have specialised Hopper kernels yet
    // 一般的模型单独调用fast_preprocess_weights_for_mixed_gemm时，shape中的num_experts=1，且force_interleave=false
    // 但是在MOE情况，num_experts>1时，force_interleave=true
    arch = 80;
  }
  LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

  KLLM_KERNEL_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

  size_t num_elts = 1;
  for (auto const& dim : shape) {
    num_elts *= dim;
  }

  const size_t num_bytes = num_elts * get_weight_quant_bits(quant_type) / 8;

  int8_t* src = row_major_quantized_weight;
  int8_t* dst = preprocessed_quantized_weight;

  // Works on row major data, so issue this permutation first.
  if (details.uses_imma_ldsm) {
    fast_permute_B_rows_for_mixed_gemm(dst, src, device_row_permutation, row_permutation_size, shape, quant_type, arch, stream);
    fast_swap(src,dst,num_bytes,stream);
  }

  if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR) {
    fast_subbyte_transpose(dst, src, shape, quant_type, stream);
    fast_swap(src,dst,num_bytes,stream);
  }

  if (details.columns_interleaved > 1) {
    fast_interleave_column_major_tensor(dst, src, shape, quant_type, details, stream);
    fast_swap(src,dst,num_bytes,stream);
  }

  if (arch >= 70 && arch < 90) {
    fast_add_bias_and_interleave_quantized_tensor_inplace(src, num_elts, quant_type, stream);
  }
  fast_swap(src,dst,num_bytes,stream);
}

}  // namespace nvidia
}  // namespace llm_kernels