/*
 * Modify from
 * https://github.com/vllm-project/vllm/blob/v0.2.3/csrc/pos_encoding_kernels.cu
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2023, The vLLM team.
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

#include <cstdint>

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

enum RotaryEmbeddingType {
  DEFAULT,
  LINEAR_SCALING,
  DYNAMIC_NTK_SCALING,
  DYNAMIC_NTK_ALPHA,
  MULTIFREQ_SCALING,
  YARN_SCALING,
  MROPE,
  INTERNLM2_DYNAMIC_NTK_SCALING
};

struct RotaryEmbeddingParam {
  int rotary_dim;
  int max_position_embeddings;
  int query_head_size;
  int key_head_size;
  int num_heads;
  int num_kv_heads;
  int64_t query_stride;
  int64_t key_stride;
  float base;
  bool is_neox;
  bool is_reverse;
  cudaStream_t stream;

  const int64_t* positions;

  // Due to the optimization of PrefixCaching for computation reuse, a mask is used during rotary_embedding
  // kernel forward to avoid multiple executions of rotary_embedding on the prefix portion.
  const int64_t* mask;
  int num_tokens_;

  RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  float scaling_factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position_embeddings = 8192;
  float scaling_alpha = 1.0f;
  const int* mrope_section = nullptr;
  // for yarn
  float beta_fast = 32.0f;
  float beta_slow = 1.0f;
  float mscale = 1.0f;
  float mscale_all_dim = 1.0f;
  bool use_deepseek_rope = false;
};

class RotaryEmbeddingCuda {
 public:
  template <typename T>
  void SetConfig(void* cos_sin_cache,  // temp buffer, [max_position_embeddings, rotary_dim]
                 const int rotary_dim, const int max_position_embeddings, const float base, const int query_head_size,
                 const int key_head_size, const int num_heads, const int num_kv_heads, const int stride_size,
                 const bool is_neox, cudaStream_t& stream,
                 const RotaryEmbeddingType rotary_embedding_type = RotaryEmbeddingType::DEFAULT,
                 const float scaling_factor = 1.0f, const float low_freq_factor = 1.0f,
                 const float high_freq_factor = 4.0f, const int original_max_position_embeddings = 8192,
                 const float scaling_alpha = 1.0f, const int* mrope_section = nullptr, const float beta_fast = 32.0f,
                 const float beta_slow = 1.0f, const float mscale = 1.0f, const float mscale_all_dim = 1.0f,
                 const bool use_deepseek_rope = false);

  void SetInput(const int64_t* positions,  // [batch_size, seq_len] or [num_tokens]
                const int64_t* mask,       // [batch_size, seq_len] or [num_tokens]
                void* query,               // [batch_size, seq_len, num_heads * head_size] or
                                           // [num_tokens, num_heads * head_size]
                void* key,                 // [batch_size, seq_len, num_kv_heads * head_size] or
                                           // [num_tokens, num_kv_heads * head_size]
                int num_tokens, cudaStream_t& stream, int64_t query_stride = 0, int64_t key_stride = 0, 
                int query_head_size = 0, int key_head_size = 0, bool is_reverse = false);

  template <typename T>
  void Forward();

 private:
  RotaryEmbeddingParam params_;
  void* cos_sin_cache_;  // [max_position_embeddings, rotary_dim]
  void* query_;
  void* key_;
};

}  // namespace nvidia
}  // namespace llm_kernels
