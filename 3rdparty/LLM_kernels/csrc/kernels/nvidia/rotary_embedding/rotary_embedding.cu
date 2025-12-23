/*
 * Modify from
 * https://github.com/vllm-project/vllm/blob/v0.2.3/csrc/pos_encoding_kernels.cu
 * https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/pos_encoding_kernels.cu
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

#include "rotary_embedding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cmath>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

template <typename T, bool IS_NEOX>
inline __device__ void ApplyRotaryEmbedding(T* __restrict__ arr, const T* __restrict__ cos_ptr,
                                            const T* __restrict__ sin_ptr, int rot_offset, int embed_dim,
                                            bool reverse = false) {
  int x_index, y_index;
  T cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = __ldg(cos_ptr + x_index);
    sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = __ldg(cos_ptr + x_index / 2);
    sin = __ldg(sin_ptr + x_index / 2);
  }
  const float cos_f = cos;
  const float sin_f = sin;
  if (reverse) {
    const float d = cos_f * cos_f + sin_f * sin_f;
    const float x = arr[x_index];
    const float y = arr[y_index];
    arr[x_index] = (x * cos_f + y * sin_f) / d;
    arr[y_index] = (y * cos_f - x * sin_f) / d;
    return;
  }
  const T x = arr[x_index];
  const T y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename T, bool IS_NEOX>
__global__ void InvokeRotaryEmbeddingKernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or [num_tokens]
    const int64_t* __restrict__ mask,       // [batch_size, seq_len] or [num_tokens]
    T* __restrict__ query,  // [batch_size, seq_len, num_heads, query_head_size] or [num_tokens, num_heads,
                            // query_head_size]
    T* __restrict__ key,    // [batch_size, seq_len, num_kv_heads, key_head_size] or [num_tokens, num_kv_heads,
                            // key_head_size]
    const T* __restrict__ cos_sin_cache,  // [max_position_embeddings, 2, rotary_dim // 2]
    const int rotary_dim, const int64_t query_stride, const int64_t key_stride, const int num_heads,
    const int num_kv_heads, const int query_head_size, const int key_head_size, const bool is_reverse) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t mask_i = mask[token_idx];
  if (mask_i == 0) {
    return;
  }

  const T* cache_ptr = cos_sin_cache + pos * rotary_dim;
  const int embed_dim = rotary_dim / 2;
  const T* cos_ptr = cache_ptr;
  const T* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * query_head_size;
    const int rot_offset = i % embed_dim;
    if (query != nullptr) {
      ApplyRotaryEmbedding<T, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, is_reverse);
    }
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * key_head_size;
    const int rot_offset = i % embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, is_reverse);
  }
}

// Multimodal Rotary Position Embedding（M-ROPE）
template <typename T, bool IS_NEOX>
__global__ void InvokeMRotaryEmbeddingKernel(
    const int64_t* __restrict__ positions,  // [3, batch_size, seq_len] or [3, num_tokens]
    const int64_t* __restrict__ mask,       // [batch_size, seq_len] or [num_tokens]
    T* __restrict__ query,  // [batch_size, seq_len, num_heads, query_head_size] or [num_tokens, num_heads,
                            // query_head_size]
    T* __restrict__ key,    // [batch_size, seq_len, num_kv_heads, key_head_size] or [num_tokens, num_kv_heads,
                            // key_head_size]
    const T* __restrict__ cos_sin_cache,  // [max_position_embeddings, 2, rotary_dim // 2]
    const int* __restrict__ mrope_section, const int rotary_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int query_head_size, const int key_head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t mask_i = mask[token_idx];
  if (mask_i == 0) {
    return;
  }

  const int embed_dim = rotary_dim / 2;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * query_head_size;
    const int rot_offset = i % embed_dim;

    // M-ROPE has three components: temporal (`positions[0]`), height (`positions[1]`),
    // and width (`positions[2]`).
    // For rotary dimention in `[0, mrope_section[0])`, use temporal position indices;
    // for rotary dimention in `[mrope_section[0], mrope_section[1])`, use height position indices;
    // and for rotary dimention in `[mrope_section[1], mrope_section[2])`, use width position indices.
    // Here `mrope_section[2]` is guaranteed to be equal to `embed_dim`.
    const int pos_index = rot_offset < mrope_section[0] ? 0 : (rot_offset < mrope_section[1] ? 1 : 2);
    const int64_t pos = positions[3 * token_idx + pos_index];
    const T* cache_ptr = cos_sin_cache + pos * rotary_dim;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * key_head_size;
    const int rot_offset = i % embed_dim;

    // M-ROPE has three components: temporal (`positions[0]`), height (`positions[1]`),
    // and width (`positions[2]`).
    // For rotary dimention in `[0, mrope_section[0])`, use temporal position indices;
    // for rotary dimention in `[mrope_section[0], mrope_section[1])`, use height position indices;
    // and for rotary dimention in `[mrope_section[1], mrope_section[2])`, use width position indices.
    // Here `mrope_section[2]` is guaranteed to be equal to `embed_dim`.
    const int pos_index = rot_offset < mrope_section[0] ? 0 : (rot_offset < mrope_section[1] ? 1 : 2);
    const int64_t pos = positions[3 * token_idx + pos_index];
    const T* cache_ptr = cos_sin_cache + pos * rotary_dim;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

// XDRope
template <typename T, bool IS_NEOX>
__global__ void InvokeXDRotaryEmbeddingKernel(
    const int64_t* __restrict__ positions,  // [4, num_tokens] 实际数据是按照[num_tokens,4]存储的
    const int64_t* __restrict__ mask,       // [num_tokens]
    T* __restrict__ query,                  // [num_tokens, num_heads, query_head_size]
    T* __restrict__ key,                    // [num_tokens, num_kv_heads, key_head_size]
    const T* __restrict__ cos_sin_cache,    // [max_position_embeddings, 2, rotary_dim // 2]
    const int* __restrict__ xdrope_section, const int rotary_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int query_head_size, const int key_head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t mask_i = mask[token_idx];
  if (mask_i == 0) {
    return;
  }

  // 四个维度分别是[seq,x,y,t]
  auto get_section = [&xdrope_section](int rot_offset) -> int {
    if (rot_offset < xdrope_section[0]) {
      return 0;
    }
    if (rot_offset < xdrope_section[1]) {
      return 1;
    }
    if (rot_offset < xdrope_section[2]) {
      return 2;
    }
    return 3;
  };

  const int embed_dim = rotary_dim / 2;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * query_head_size;
    const int pos_index = get_section(rot_offset);
    const int64_t pos = positions[token_idx * 4 + pos_index];
    const T* cache_ptr = cos_sin_cache + pos * rotary_dim;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * key_head_size;
    const int pos_index = get_section(rot_offset);
    const int64_t pos = positions[token_idx * 4 + pos_index];
    const T* cache_ptr = cos_sin_cache + pos * rotary_dim;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    ApplyRotaryEmbedding<T, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}

#define DISPATCH_ROTARY_EMBEDDING_BY_NEOX(ROTARY_EMBEDDING, IS_NEOX, ...)                       \
  if (IS_NEOX) {                                                                                \
    Invoke##ROTARY_EMBEDDING##Kernel<T, true><<<grid, block, 0, params.stream>>>(__VA_ARGS__);  \
  } else {                                                                                      \
    Invoke##ROTARY_EMBEDDING##Kernel<T, false><<<grid, block, 0, params.stream>>>(__VA_ARGS__); \
  }

template <typename T>
void LaunchRotaryEmbedding(const RotaryEmbeddingParam& params, T* cos_sin_cache, T* query, T* key) {
  dim3 grid(params.num_tokens_);
  dim3 block(std::min(params.num_heads * params.rotary_dim / 2, 512));

  if (params.mrope_section != nullptr) {
    DISPATCH_ROTARY_EMBEDDING_BY_NEOX(MRotaryEmbedding, params.is_neox, params.positions, params.mask, query, key,
                                      cos_sin_cache, params.mrope_section, params.rotary_dim, params.query_stride,
                                      params.key_stride, params.num_heads, params.num_kv_heads, params.query_head_size,
                                      params.key_head_size);
  } else if (params.xdrope_section != nullptr) {
    KLLM_KERNEL_CHECK(params.rotary_dim == params.query_head_size);
    KLLM_KERNEL_CHECK(params.query_head_size == params.key_head_size);
    DISPATCH_ROTARY_EMBEDDING_BY_NEOX(XDRotaryEmbedding, params.is_neox, params.positions, params.mask, query, key,
                                      cos_sin_cache, params.xdrope_section, params.rotary_dim, params.query_stride,
                                      params.key_stride, params.num_heads, params.num_kv_heads, params.query_head_size,
                                      params.key_head_size);
  } else {
    DISPATCH_ROTARY_EMBEDDING_BY_NEOX(RotaryEmbedding, params.is_neox, params.positions, params.mask, query, key,
                                      cos_sin_cache, params.rotary_dim, params.query_stride, params.key_stride,
                                      params.num_heads, params.num_kv_heads, params.query_head_size,
                                      params.key_head_size, params.is_reverse);
  }
}

template void LaunchRotaryEmbedding<float>(const RotaryEmbeddingParam& params, float* cos_sin_cache, float* query,
                                           float* key);
template void LaunchRotaryEmbedding<half>(const RotaryEmbeddingParam& params, half* cos_sin_cache, half* query,
                                          half* key);
template void LaunchRotaryEmbedding<__nv_bfloat16>(const RotaryEmbeddingParam& params, __nv_bfloat16* cos_sin_cache,
                                                   __nv_bfloat16* query, __nv_bfloat16* key);

template <typename T>
__global__ void InvokeComputeCosSinWithCacheKernel(T* __restrict__ cos_sin_cache, const int rotary_dim,
                                                   const int max_position_embeddings, const float base,
                                                   const float scaling) {
  int pos = blockIdx.x;
  for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
    float inv_freq = 1.0 / pow(base, rid * 2 / (float)rotary_dim);
    float freq = pos * inv_freq / scaling;
    cos_sin_cache[pos * rotary_dim + rid] = (T)cos(freq);
    cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = (T)sin(freq);
  }
}

template <typename T>
__global__ void InvokeComputeMultiFreqCosSinWithCacheKernel(T* __restrict__ cos_sin_cache, const int rotary_dim,
                                                            const int max_position_embeddings, const float base,
                                                            const float scaling, const float low_freq_factor,
                                                            const float high_freq_factor,
                                                            const int original_max_position_embeddings) {
  int pos = blockIdx.x;
  float low_freq_wavelen = (float)original_max_position_embeddings / low_freq_factor;
  float high_freq_wavelen = (float)original_max_position_embeddings / high_freq_factor;
  for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
    float inv_freq = 1.0f / pow(base, rid * 2 / (float)rotary_dim);
    float wavelen = 2.0f * M_PI / inv_freq;
    float freq = inv_freq;
    // Same logic as :
    // https://github.com/vllm-project/vllm/blob/c5df56f88bc8a5a32a0534793f48182a333aeca4/vllm/model_executor/layers/rotary_embedding.py#L742
    if (wavelen < high_freq_wavelen) {
      freq = inv_freq;
    } else if (wavelen > low_freq_wavelen) {
      freq = inv_freq / scaling;
    } else {
      float smooth =
          ((float)original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
      freq = (1.0f - smooth) * inv_freq / scaling + smooth * inv_freq;
    }
    freq = pos * freq;
    cos_sin_cache[pos * rotary_dim + rid] = (T)cos(freq);
    cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = (T)sin(freq);
  }
}

template <typename T>
__global__ void InvokeYarnComputeCosSinWithCacheKernel(T* __restrict__ cos_sin_cache, const int rotary_dim,
                                                       const int max_position_embeddings, const float base,
                                                       const float scaling, const float mscale, const float beta_fast,
                                                       const float beta_slow) {
  // Same logic as :
  // https://github.com/vllm-project/vllm/blob/9f669a9a7c2b2d0a7963a6e29253280e57680adb/vllm/model_executor/layers/rotary_embedding.py#L215
  int pos = blockIdx.x;
  float extrapolation_factor = 1.0f;

  float low_dim = ((float)rotary_dim * std::log((float)max_position_embeddings / (beta_fast * 2.0f * M_PI))) /
                  (2.0f * std::log(base));
  float low = std::floor(low_dim);
  low = std::max(low, 0.0f);
  float high_dim = ((float)rotary_dim * std::log((float)max_position_embeddings / (beta_slow * 2.0f * M_PI))) /
                   (2.0f * std::log(base));
  float high = std::ceil(high_dim);
  high = std::min(high, (float)rotary_dim - 1.0f);
  for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
    float pos_freqs = pow(base, rid * 2 / (float)rotary_dim);
    float inv_freq_extrapolation = 1.0f / pos_freqs;
    float inv_freq_interpolation = 1.0f / (scaling * pos_freqs);
    float ramp_mask;
    if (low == high) {
      ramp_mask = ((float)rid - low) / (high + 0.001f - low);
    } else {
      ramp_mask = ((float)rid - low) / (high - low);
    }
    if (ramp_mask <= 0.0f) {
      ramp_mask = 0.0f;
    }
    if (ramp_mask >= 1.0f) {
      ramp_mask = 1.0f;
    }
    float inv_freq_mask = (1.0f - ramp_mask) * extrapolation_factor;
    float inv_freq = inv_freq_interpolation * (1.0f - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;

    float freq = pos * inv_freq;
    cos_sin_cache[pos * rotary_dim + rid] = (T)(cos(freq) * mscale);
    cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = (T)(sin(freq) * mscale);
  }
}

template <typename T>
__global__ void InvokeXDRopeComputeCosSinWithCacheKernel(T* __restrict__ cos_sin_cache, const int rotary_dim,
                                                         const int max_position_embeddings, const float base,
                                                         const float scaling_alpha) {
  int pos = blockIdx.x;
  float scaled_base = base * powf(scaling_alpha, (float)rotary_dim / ((float)rotary_dim - 2.0f));
  for (int rid = threadIdx.x; rid < rotary_dim / 2; rid += blockDim.x) {
    float inv_freq = 1.0f / powf(scaled_base, (rid * 2) / (float)rotary_dim);
    float freq = pos * inv_freq;
    cos_sin_cache[pos * rotary_dim + rid] = (T)cosf(freq);
    cos_sin_cache[pos * rotary_dim + rotary_dim / 2 + rid] = (T)sinf(freq);
  }
}

template <typename T>
void ComputeCosSinWithCache(const RotaryEmbeddingParam& params, T* cos_sin_cache) {
  size_t extend_max_len = params.max_position_embeddings;
  dim3 block(std::min(params.rotary_dim / 2, DEFAULT_CUDA_BLOCK_THREADS_NUM));

  float base = params.base;
  float scaling = 1.0f;
  // Same logic as :
  // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
  if (params.rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
    extend_max_len = params.max_position_embeddings * params.scaling_factor;
    base = std::pow(params.base * ((params.scaling_factor * extend_max_len / params.max_position_embeddings) -
                                   (params.scaling_factor - 1)),
                    (params.rotary_dim / (params.rotary_dim - 2)));
  }
  // InternLM2 use InternLM2RotaryEmbedding, Same logic as :
  // https://huggingface.co/internlm/internlm2_5-7b-chat/blob/main/modeling_internlm2.py#L111
  if (params.rotary_embedding_type == RotaryEmbeddingType::INTERNLM2_DYNAMIC_NTK_SCALING) {
    extend_max_len = params.max_position_embeddings * params.scaling_factor;
    base = params.base;
  }
  if (params.rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_ALPHA) {
    base = params.base * std::pow(params.scaling_alpha, params.rotary_dim / (params.rotary_dim - 2));
  }
  if (params.rotary_embedding_type == RotaryEmbeddingType::LINEAR_SCALING) {
    extend_max_len = params.max_position_embeddings * params.scaling_factor;
    scaling = params.scaling_factor;
  }
  if (params.rotary_embedding_type == RotaryEmbeddingType::MULTIFREQ_SCALING) {
    scaling = params.scaling_factor;
    float low_freq_factor = params.low_freq_factor;
    float high_freq_factor = params.high_freq_factor;
    int original_max_position_embeddings = params.original_max_position_embeddings;

    dim3 grid(extend_max_len);
    InvokeComputeMultiFreqCosSinWithCacheKernel<T>
        <<<grid, block, 0, params.stream>>>(cos_sin_cache, params.rotary_dim, extend_max_len, base, scaling,
                                            low_freq_factor, high_freq_factor, original_max_position_embeddings);
  } else if (params.rotary_embedding_type == RotaryEmbeddingType::XDROPE) {
    extend_max_len = params.max_position_embeddings * params.scaling_alpha;
    dim3 grid(extend_max_len);
    InvokeXDRopeComputeCosSinWithCacheKernel<T><<<grid, block, 0, params.stream>>>(
        cos_sin_cache, params.rotary_dim, extend_max_len, params.base, params.scaling_alpha);
  } else if (params.rotary_embedding_type == RotaryEmbeddingType::YARN_SCALING) {
    scaling = params.scaling_factor;
    int original_max_position_embeddings = params.original_max_position_embeddings;
    extend_max_len = original_max_position_embeddings * scaling;
    float mscale = params.mscale;
    float beta_fast = params.beta_fast;
    float beta_slow = params.beta_slow;
    dim3 grid(extend_max_len);
    InvokeYarnComputeCosSinWithCacheKernel<T><<<grid, block, 0, params.stream>>>(cos_sin_cache, params.rotary_dim,
                                                                                 original_max_position_embeddings, base,
                                                                                 scaling, mscale, beta_fast, beta_slow);
  } else {
    dim3 grid(extend_max_len);
    InvokeComputeCosSinWithCacheKernel<T>
        <<<grid, block, 0, params.stream>>>(cos_sin_cache, params.rotary_dim, extend_max_len, base, scaling);
  }
}

template void ComputeCosSinWithCache<float>(const RotaryEmbeddingParam& params, float* cos_sin_cache);
template void ComputeCosSinWithCache<half>(const RotaryEmbeddingParam& params, half* cos_sin_cache);
template void ComputeCosSinWithCache<__nv_bfloat16>(const RotaryEmbeddingParam& params, __nv_bfloat16* cos_sin_cache);

void RotaryEmbeddingCuda::SetInput(const int64_t* positions,  // [batch_size, seq_len] or [num_tokens]
                                   const int64_t* mask,
                                   void* query,  // [batch_size, seq_len, num_heads * query_head_size] or
                                                 // [num_tokens, num_heads * query_head_size]
                                   void* key,    // [batch_size, seq_len, num_kv_heads * key_head_size] or
                                                 // [num_tokens, num_kv_heads * key_head_size]
                                   int num_tokens, cudaStream_t& stream, int64_t query_stride, int64_t key_stride,
                                   int query_head_size, int key_head_size, bool is_reverse) {
  params_.positions = positions;
  params_.mask = mask;
  query_ = query;
  key_ = key;
  params_.num_tokens_ = num_tokens;
  params_.is_reverse = is_reverse;
  params_.stream = stream;
  if (query_stride > 0) {
    params_.query_stride = query_stride;
  }
  if (key_stride > 0) {
    params_.key_stride = key_stride;
  }
  if (query_head_size > 0) {
    params_.query_head_size = query_head_size;
  }
  if (key_head_size > 0) {
    params_.key_head_size = key_head_size;
  }
}

template <typename T>
void RotaryEmbeddingCuda::Forward() {
  LaunchRotaryEmbedding<T>(params_, reinterpret_cast<T*>(cos_sin_cache_), reinterpret_cast<T*>(query_),
                           reinterpret_cast<T*>(key_));
}

template void RotaryEmbeddingCuda::Forward<float>();
template void RotaryEmbeddingCuda::Forward<half>();
template void RotaryEmbeddingCuda::Forward<__nv_bfloat16>();

template <typename T>
void RotaryEmbeddingCuda::SetConfig(void* cos_sin_cache, const int rotary_dim, const int max_position_embeddings,
                                    const float base, const int query_head_size, const int key_head_size,
                                    const int num_heads, const int num_kv_heads, const int stride_size,
                                    const bool is_neox, cudaStream_t& stream,
                                    const RotaryEmbeddingType rotary_embedding_type, const float scaling_factor,
                                    const float low_freq_factor, const float high_freq_factor,
                                    const int original_max_position_embeddings, const float scaling_alpha,
                                    const int* mrope_section, const int* xdrope_section, const float beta_fast,
                                    const float beta_slow, const float mscale, const float mscale_all_dim,
                                    const bool use_deepseek_rope) {
  cos_sin_cache_ = cos_sin_cache;
  params_.rotary_dim = rotary_dim;
  params_.max_position_embeddings = max_position_embeddings;
  params_.base = base;
  params_.query_head_size = query_head_size;
  params_.key_head_size = key_head_size;
  params_.num_heads = num_heads;
  params_.num_kv_heads = num_kv_heads;
  params_.is_neox = is_neox;
  params_.stream = stream;
  params_.query_stride = stride_size;
  params_.key_stride = stride_size;
  params_.rotary_embedding_type = rotary_embedding_type;
  params_.scaling_factor = scaling_factor;
  params_.low_freq_factor = low_freq_factor;
  params_.high_freq_factor = high_freq_factor;
  params_.original_max_position_embeddings = original_max_position_embeddings;
  params_.scaling_alpha = scaling_alpha;
  params_.mrope_section = mrope_section;
  params_.xdrope_section = xdrope_section;
  params_.beta_fast = beta_fast;
  params_.beta_slow = beta_slow;
  params_.mscale = mscale;
  params_.mscale_all_dim = mscale_all_dim;
  params_.use_deepseek_rope = use_deepseek_rope;
  if (use_deepseek_rope) {
    params_.query_stride = num_heads * rotary_dim;
    params_.key_stride = rotary_dim;
    params_.num_kv_heads = 1;
    params_.query_head_size = rotary_dim;
    params_.key_head_size = rotary_dim;
  }
  ComputeCosSinWithCache<T>(params_, reinterpret_cast<T*>(cos_sin_cache_));
}

template void RotaryEmbeddingCuda::SetConfig<float>(
    void* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
    const int query_head_size, const int key_head_size, const int num_heads, const int num_kv_heads,
    const int stride_size, const bool is_neox, cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type,
    const float scaling_factor, const float low_freq_factor, const float high_freq_factor,
    const int original_max_position_embeddings, const float scaling_alpha, const int* mrope_section,
    const int* xdrope_section, const float beta_fast, const float beta_slow, const float mscale,
    const float mscale_all_dim, const bool use_deepseek_rope);
template void RotaryEmbeddingCuda::SetConfig<half>(
    void* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
    const int query_head_size, const int key_head_size, const int num_heads, const int num_kv_heads,
    const int stride_size, const bool is_neox, cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type,
    const float scaling_factor, const float low_freq_factor, const float high_freq_factor,
    const int original_max_position_embeddings, const float scaling_alpha, const int* mrope_section,
    const int* xdrope_section, const float beta_fast, const float beta_slow, const float mscale,
    const float mscale_all_dim, const bool use_deepseek_rope);
template void RotaryEmbeddingCuda::SetConfig<__nv_bfloat16>(
    void* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
    const int query_head_size, const int key_head_size, const int num_heads, const int num_kv_heads,
    const int stride_size, const bool is_neox, cudaStream_t& stream, const RotaryEmbeddingType rotary_embedding_type,
    const float scaling_factor, const float low_freq_factor, const float high_freq_factor,
    const int original_max_position_embeddings, const float scaling_alpha, const int* mrope_section,
    const int* xdrope_section, const float beta_fast, const float beta_slow, const float mscale,
    const float mscale_all_dim, const bool use_deepseek_rope);

}  // namespace nvidia
}  // namespace llm_kernels
