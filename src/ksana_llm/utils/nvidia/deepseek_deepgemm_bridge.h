/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <torch/torch.h>
#include <optional>

namespace deepseek {

void InitializeDeepGemm(int num_sms);

void ShutdownDeepGemmRuntime();

void SmxxFp8MqaLogits(const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_scales,
                      const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                      const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits, int seq_len, int seq_len_kv,
                      int stride_kv, int num_heads, int head_dim, int seq_len_alignment, bool enable_clean);

void SmxxFp8PagedMqaLogits(const torch::Tensor& q, const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_scales,
                           const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& logits,
                           const torch::Tensor& block_table, const torch::Tensor& schedule_meta, int batch_size,
                           int next_n, int num_heads, int head_dim, int num_kv_blocks, int block_kv,
                           int kv_cache_stride_bytes, int logits_stride, int block_table_stride, int num_sms,
                           int num_math_warp_groups, bool enable_clean, int clean_max_context_len);

void SmxxPagedMqaLogitsMetadata(const torch::Tensor& context_lens, const torch::Tensor& schedule_metadata,
                                int batch_size, int block_kv, int num_sms);

}  // namespace deepseek
