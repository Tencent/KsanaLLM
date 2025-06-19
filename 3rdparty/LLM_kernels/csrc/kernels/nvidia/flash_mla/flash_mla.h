/*
 * Copyright (c) 2025 DeepSeek
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Adapted from
 * [FlashMLA Project] https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/flash_api.cpp
 */

#include <cuda_runtime.h>
#include <cstdint>

#pragma once

namespace llm_kernels {
namespace nvidia {

struct FlashMlaWorkspaceMap {
  int *tile_scheduler_metadata_ptr;
  int *num_splits_ptr;
  float *softmax_lse_ptr;
  float *softmax_lse_accum_ptr;
  float *out_accum_ptr;
  int num_sm_parts = 1;
};

void InvokeGetMlaMetadata(int *b_seqlen, FlashMlaWorkspaceMap &workspace_param, int tokens_num, cudaStream_t stream);
void GetNumSmParts(FlashMlaWorkspaceMap &workspace_param, const int num_heads_per_head_k, const int num_heads_k,
                   int rank, cudaStream_t stream);

template <typename T>
void InvokeFlashMla(T *q, T *k_buffer, int q_len, float sm_scale, void *block_table_ptr, void *b_seqlen,
                    void *tile_scheduler_metadata_ptr, void *num_splits_ptr, void *workspace, void *att_out,
                    int batch_size, int num_heads, int kv_lora_rank, int qk_rope_head_dim, int page_size,
                    int max_blocks_per_seq, int rank, size_t block_num, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
