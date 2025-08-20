/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T, bool DO_POSITION_ENCODING, bool USE_EMB_SCALE>
__global__ void LookupFusedEmbeddingKernelWithCSRKernel(T* output_hidden_units, const T* embedding_table,
                                                        const T* pos_table, const T emb_scale,
                                                        InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                                        const int32_t* input_ids, const size_t* steps,
                                                        const size_t* ids_offsets, const size_t* prefix_offsets,
                                                        const int32_t batch_size, const uint32_t hidden_units,
                                                        const size_t vocab_size, const size_t vocab_id) {
  // NOTE(karlluo): config of grid and block
  // grid: (min(batch_size, 65536), 32);
  // block: (min(hidden_units, 512));
  size_t input_ids_idx_offset = ids_offsets[blockIdx.x] - prefix_offsets[blockIdx.x];
  size_t next_ids_offset = ids_offsets[blockIdx.x + 1] - prefix_offsets[blockIdx.x + 1];
  size_t ids_num = next_ids_offset - input_ids_idx_offset;
  size_t step = 0ul;

  // TODO(karlluo): optimization unroll ?
#pragma unroll
  for (uint32_t token_id = blockIdx.y; token_id < ids_num; token_id += gridDim.y) {
    int32_t real_token_id = input_ids[input_ids_idx_offset + token_id];
    // on each GPU, emb range is [vocab_id * vocab_size, (vocab_id + 1) * vocab_size)
    // read_id bigger than the vocabulary size, handle next id
    if (real_token_id >= ((vocab_id + 1) * vocab_size) || real_token_id < (vocab_id * vocab_size)) {
      continue;
    }
    if constexpr (DO_POSITION_ENCODING) {
      step = steps[input_ids_idx_offset + token_id];
    }

    // copy emb vec value
    // TODO(karlluo): optimization veterize ?
#pragma unroll
    for (uint32_t emb_id = threadIdx.x; emb_id < hidden_units; emb_id += blockDim.x) {
      T emb_vec_val = embedding_table[(real_token_id - vocab_id * vocab_size) * hidden_units + emb_id];
      // NOTE(karlluo): half has not static cast in kernel
      if constexpr (DO_POSITION_ENCODING) {
        T pos_emb_vec_val = pos_table[step * hidden_units + emb_id];
        output_hidden_units[(input_ids_idx_offset + token_id) * hidden_units + emb_id] =
            emb_vec_val * emb_scale + pos_emb_vec_val;
      } else if constexpr (USE_EMB_SCALE) {
        output_hidden_units[(input_ids_idx_offset + token_id) * hidden_units + emb_id] = emb_vec_val * emb_scale;
      } else {
        output_hidden_units[(input_ids_idx_offset + token_id) * hidden_units + emb_id] = emb_vec_val;
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
