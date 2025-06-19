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
 * [FlashMLA Project] https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kernels/params.h
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace llm_kernels {
namespace nvidia {
struct Flash_fwd_mla_params {
  using index_t = int64_t;

  int b;  // batch size
  int s_q;
  int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
  int d, d_v;         // K/V dimension
  int h_q, h_k;       // The number of Q/K heads
  int num_blocks;     // Number of blocks in total
  int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
  bool is_causal;
  float scale_softmax, scale_softmax_log2;

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ o_ptr;
  void *__restrict__ softmax_lse_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t o_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t o_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t o_head_stride;

  int *__restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;
  int *__restrict__ seqlens_k_ptr;

  int *__restrict__ tile_scheduler_metadata_ptr;
  int num_sm_parts;
  int *__restrict__ num_splits_ptr;

  int total_num_splits;
  void *__restrict__ softmax_lseaccum_ptr;
  void *__restrict__ oaccum_ptr;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

struct Mla_metadata_params {
  int *__restrict__ seqlens_k_ptr;
  int *__restrict__ tile_scheduler_metadata_ptr;
  int *__restrict__ num_splits_ptr;
  int batch_size;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int num_sm_parts;
};
}  // namespace nvidia
}  // namespace llm_kernels