/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

// Optimized fused embedding lookup with optional position encoding and scaling.
// Key optimizations implemented here:
// 1) Warp-level broadcast: input token id (and step) are loaded once per warp and
//    distributed with __shfl_sync, avoiding redundant scalar loads per thread.
// 2) 16-byte vectorized IO: use float4 for float and uint4 for half/bfloat16 to move
//    16B per transaction, reducing instruction count and improving memory throughput.
template <typename T, bool DO_POSITION_ENCODING, bool USE_EMB_SCALE>
__global__ void LookupFusedEmbeddingKernelWithCSRKernel(T* __restrict__ output_hidden_units,
                                                        const T* __restrict__ embedding_table,
                                                        const T* __restrict__ pos_table, const T emb_scale,
                                                        InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                                        const int32_t* __restrict__ input_ids,
                                                        const size_t* __restrict__ steps,
                                                        const size_t* __restrict__ ids_offsets,
                                                        const size_t* __restrict__ prefix_offsets,
                                                        const int32_t batch_size, const uint32_t hidden_units,
                                                        const size_t vocab_size, const size_t vocab_id) {
  // NOTE(karlluo): config of grid and block
  // grid: (min(batch_size, 65536), 32);
  // block: (min(hidden_units, 512));
  size_t input_ids_idx_offset = ids_offsets[blockIdx.x] - prefix_offsets[blockIdx.x];
  size_t next_ids_offset = ids_offsets[blockIdx.x + 1] - prefix_offsets[blockIdx.x + 1];
  size_t ids_num = next_ids_offset - input_ids_idx_offset;
  const unsigned full_mask = 0xffffffffu;

#pragma unroll
  for (uint32_t token_id = blockIdx.y; token_id < ids_num; token_id += gridDim.y) {
    // 1) Warp-level broadcast of the current token id (and optional position step).
    //    Only lane 0 performs global loads; the rest of the warp gets values via shfl.
    int32_t real_token_id = 0;
    uint32_t position_step_index = 0u;
    constexpr int32_t float_byte_num = 4;
    constexpr uint32_t warp_id_selection_mask = 31u;
    // Only lane 0 performs global loads; the rest of the warp gets values via shfl.
    // reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#broadcast-of-a-single-value-across-a-warp
    if ((threadIdx.x & warp_id_selection_mask) == 0) {
      real_token_id = input_ids[input_ids_idx_offset + token_id];
      if constexpr (DO_POSITION_ENCODING) {
        position_step_index = static_cast<uint32_t>(steps[input_ids_idx_offset + token_id]);
      }
    }
    real_token_id = __shfl_sync(full_mask, real_token_id, 0);
    if constexpr (DO_POSITION_ENCODING) {
      position_step_index = __shfl_sync(full_mask, position_step_index, 0);
    }
    // on each GPU, emb range is [vocab_id * vocab_size, (vocab_id + 1) * vocab_size)
    // read_id bigger than the vocabulary size, handle next id
    if (real_token_id >= ((vocab_id + 1) * vocab_size) || real_token_id < (vocab_id * vocab_size)) {
      continue;
    }
    // 2) Precompute base offsets for this token once.
    const size_t emb_row_base = static_cast<size_t>(real_token_id - static_cast<int32_t>(vocab_id * vocab_size)) *
                                static_cast<size_t>(hidden_units);
    const size_t token_out_base = (input_ids_idx_offset + token_id) * static_cast<size_t>(hidden_units);
    const size_t pos_base = static_cast<size_t>(position_step_index) * static_cast<size_t>(hidden_units);
    constexpr uint32_t pack_four_elements = 4u;
    constexpr uint32_t pack_eight_elements = 8u;

    // For float type, we use float4 to read/write 4 elements (16B) per iteration.
    if constexpr (sizeof(T) == float_byte_num) {
      const uint32_t vectorized_loop_end = hidden_units - (hidden_units % pack_four_elements);
      // 3) float path: read/write 4 elements (16B) per iteration via float4.
      for (uint32_t hidden_offset = threadIdx.x * pack_four_elements; hidden_offset < vectorized_loop_end;
            hidden_offset += blockDim.x * pack_four_elements) {
        const float4* emb_v = reinterpret_cast<const float4*>(embedding_table + emb_row_base + hidden_offset);
        float4 emb_pack = emb_v[0];
        float4 out_pack;
        if constexpr (DO_POSITION_ENCODING) {
          const float4* pos_v = reinterpret_cast<const float4*>(pos_table + pos_base + hidden_offset);
          float4 pos_pack = pos_v[0];
          out_pack.x = emb_pack.x * static_cast<float>(emb_scale) + pos_pack.x;
          out_pack.y = emb_pack.y * static_cast<float>(emb_scale) + pos_pack.y;
          out_pack.z = emb_pack.z * static_cast<float>(emb_scale) + pos_pack.z;
          out_pack.w = emb_pack.w * static_cast<float>(emb_scale) + pos_pack.w;
        } else if constexpr (USE_EMB_SCALE) {
          out_pack.x = emb_pack.x * static_cast<float>(emb_scale);
          out_pack.y = emb_pack.y * static_cast<float>(emb_scale);
          out_pack.z = emb_pack.z * static_cast<float>(emb_scale);
          out_pack.w = emb_pack.w * static_cast<float>(emb_scale);
        } else {
          out_pack = emb_pack;
        }
        float4* out_v = reinterpret_cast<float4*>(output_hidden_units + token_out_base + hidden_offset);
        out_v[0] = out_pack;
      }
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) {
      // For half/bfloat16 type, we use uint4 to read/write 8 elements (16B) per iteration.
      // Check if base addresses are 16-byte aligned for vectorized access
      const bool emb_aligned = (reinterpret_cast<uintptr_t>(embedding_table + emb_row_base) % 16 == 0);
      const bool out_aligned = (reinterpret_cast<uintptr_t>(output_hidden_units + token_out_base) % 16 == 0);
      const bool pos_aligned = !DO_POSITION_ENCODING || (reinterpret_cast<uintptr_t>(pos_table + pos_base) % 16 == 0);
      if (emb_aligned && out_aligned && pos_aligned) {
        const uint32_t vectorized_loop_end = hidden_units - (hidden_units % pack_eight_elements);
        // 3) half/bfloat16 path: move 16B using uint4; compute element-wise in registers.
        //  - Each loop iteration processes one contiguous 16B vector (8 elements).
        //  - hidden_offset starts at threadIdx.x * pack so warp lanes touch disjoint, adjacent
        //      16B segments -> naturally coalesced and aligned accesses.
        //  - Step by blockDim.x * pack to keep per-thread work on whole vectors,
        //      minimizing index arithmetic.
        for (uint32_t hidden_offset = threadIdx.x * pack_eight_elements; hidden_offset < vectorized_loop_end;
              hidden_offset += blockDim.x * pack_eight_elements) {
          // Load 16B from embedding row as one transaction.
          const uint4* emb_v = reinterpret_cast<const uint4*>(embedding_table + emb_row_base + hidden_offset);
          uint4 emb_pack = emb_v[0];
          // Start output from input pack and modify elements in registers.
          uint4 out_pack = emb_pack;
          // Reinterpret both 16B vectors as arrays of 8 elements of type T to
          // perform per-element math while still committing as a single 16B store.
          T* out_elems = reinterpret_cast<T*>(&out_pack);
          const T* emb_elems = reinterpret_cast<const T*>(&emb_pack);

          if constexpr (DO_POSITION_ENCODING) {
            // out = emb * emb_scale + pos, vectorized by 16B
            const uint4* pos_v = reinterpret_cast<const uint4*>(pos_table + pos_base + hidden_offset);
            uint4 pos_pack = pos_v[0];
            const T* pos_elems = reinterpret_cast<const T*>(&pos_pack);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              out_elems[i] = emb_elems[i] * emb_scale + pos_elems[i];
            }
          } else if constexpr (USE_EMB_SCALE) {
            // out = emb * emb_scale, vectorized by 16B
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              out_elems[i] = emb_elems[i] * emb_scale;
            }
          }
          // Store back 16B in one coalesced transaction.
          uint4* out_v = reinterpret_cast<uint4*>(output_hidden_units + token_out_base + hidden_offset);
          out_v[0] = out_pack;
        }
      } else {
        // Fall back to full scalar path if not aligned - process all elements
        for (uint32_t hidden_offset = threadIdx.x; hidden_offset < hidden_units; hidden_offset += blockDim.x) {
          T emb_vec_val = embedding_table[emb_row_base + hidden_offset];
          if constexpr (DO_POSITION_ENCODING) {
            T pos_emb_vec_val = pos_table[pos_base + hidden_offset];
            output_hidden_units[token_out_base + hidden_offset] = emb_vec_val * emb_scale + pos_emb_vec_val;
          } else if constexpr (USE_EMB_SCALE) {
            output_hidden_units[token_out_base + hidden_offset] = emb_vec_val * emb_scale;
          } else {
            output_hidden_units[token_out_base + hidden_offset] = emb_vec_val;
          }
        }
      }
      // Skip tail processing - either vectorized path handled aligned portion or scalar fallback handled all
      continue;
    } else {
      // Unsupported dtype for vectorized path; log once and fall back to scalar computation.
      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("LookupFusedEmbeddingKernelWithCSRKernel: dtype not supported yet for vectorized path.\n");
      }
      for (uint32_t emb_id = threadIdx.x; emb_id < hidden_units; emb_id += blockDim.x) {
        T emb_vec_val = embedding_table[emb_row_base + emb_id];
        if constexpr (DO_POSITION_ENCODING) {
          T pos_emb_vec_val = pos_table[pos_base + emb_id];
          output_hidden_units[token_out_base + emb_id] = emb_vec_val * emb_scale + pos_emb_vec_val;
        } else if constexpr (USE_EMB_SCALE) {
          output_hidden_units[token_out_base + emb_id] = emb_vec_val * emb_scale;
        } else {
          output_hidden_units[token_out_base + emb_id] = emb_vec_val;
        }
      }
      continue;
    }

    const uint32_t scalar_start = (sizeof(T) == float_byte_num) ? (hidden_units - (hidden_units % pack_four_elements))
                                      : ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
                                             ? (hidden_units - (hidden_units % pack_eight_elements)) : 0u);
    for (uint32_t hidden_offset = scalar_start + threadIdx.x; hidden_offset < hidden_units; hidden_offset += blockDim.x) {
      T emb_vec_val = embedding_table[emb_row_base + hidden_offset];
      if constexpr (DO_POSITION_ENCODING) {
        T pos_emb_vec_val = pos_table[pos_base + hidden_offset];
        output_hidden_units[token_out_base + hidden_offset] = emb_vec_val * emb_scale + pos_emb_vec_val;
      } else if constexpr (USE_EMB_SCALE) {
        output_hidden_units[token_out_base + hidden_offset] = emb_vec_val * emb_scale;
      } else {
        output_hidden_units[token_out_base + hidden_offset] = emb_vec_val;
      }
    }
  }
}

template <typename T, bool DO_POSITION_ENCODING, bool USE_EMB_SCALE>
void LookupFusedEmbeddingWithCSRInputs(T* output_hidden_units, const T* embedding_table, const T* pos_table,
                                       const T emb_scale,
                                       InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                       const int32_t* input_ids, const size_t* steps, const size_t* ids_offsets,
                                       const size_t* prefix_offsets, const int32_t batch_size,
                                       const uint32_t hidden_units, const size_t vocab_size, const size_t vocab_id,
                                       cudaStream_t stream) {
  // each block handle one sample among batch's token last hidden units
  constexpr int32_t tokens_stride = 32;
  dim3 grid(min(static_cast<int32_t>(batch_size), DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM), tokens_stride);
  dim3 block(min(hidden_units, DEFAULT_CUDA_BLOCK_THREADS_NUM));
  LookupFusedEmbeddingKernelWithCSRKernel<T, DO_POSITION_ENCODING, USE_EMB_SCALE>
    <<<grid, block, 0, stream>>>(output_hidden_units, embedding_table, pos_table, emb_scale,
                                  prompt_param, input_ids, steps, ids_offsets, prefix_offsets, batch_size, hidden_units,
                                  vocab_size, vocab_id);
}

#define INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(T, DO_POSITION_ENCODING, USE_EMB_SCALE)                   \
  template void LookupFusedEmbeddingWithCSRInputs<T, DO_POSITION_ENCODING, USE_EMB_SCALE>(                           \
      T * output_hidden_units, const T* embedding_table, const T* pos_table, const T emb_scale,                      \
      InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param, const int32_t* input_ids, const size_t* steps,   \
      const size_t* ids_offsets, const size_t* prefix_offsets, const int32_t batch_size, const uint32_t hidden_units, \
      const size_t vocab_size, const size_t vocab_id, cudaStream_t stream);

INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(float, true, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(float, true, false);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(float, false, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(float, false, false);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(half, true, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(half, true, false);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(half, false, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(half, false, false);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(__nv_bfloat16, true, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(__nv_bfloat16, true, false);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(__nv_bfloat16, false, true);
INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS(__nv_bfloat16, false, false);

#undef INSTANTIATE_LOOKUP_FUSED_EMBEDDING_WITH_CSR_INPUTS

}  // namespace nvidia
}  // namespace llm_kernels
