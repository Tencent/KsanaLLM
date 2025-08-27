// Copyright 2024 Tencent Inc.  All rights reserved.

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <vector>

#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaRotaryEmbeddingTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  int rotary_dim{128};
  int max_position_embeddings{2048};
  int num_tokens{512};
  int num_kv_heads{40};
  int num_heads{40};
  int head_size{128};
  float base{10000};
  bool is_neox{true};
  size_t batch_size{1ul};
  RotaryEmbeddingType rotary_embedding_type{RotaryEmbeddingType::DEFAULT};
  float scaling_factor{1.0f};
  int num_iterations{100};

  template <typename T>
  void RotaryEmbeddingOnCPU(T* cos_sin_cache, const int rotary_dim, const int max_position_embeddings, const float base,
                            const bool is_neox, const RotaryEmbeddingType rotary_type = RotaryEmbeddingType::DEFAULT,
                            const float scaling_factor = 1.0f) {
    for (int pos = 0; pos < max_position_embeddings; ++pos) {
      for (int i = 0; i < rotary_dim / 2; ++i) {
        float inv_freq;
        if (rotary_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
          // Match the GPU implementation in ComputeCosSinWithCache
          float extend_max_len = max_position_embeddings * scaling_factor;
          float adjusted_base = powf(base * ((scaling_factor * extend_max_len / max_position_embeddings) -
                                           (scaling_factor - 1)),
                                    (rotary_dim / (rotary_dim - 2)));
          inv_freq = 1.0f / powf(adjusted_base, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        } else if (rotary_type == RotaryEmbeddingType::LINEAR_SCALING) {
          inv_freq = 1.0f / powf(base, static_cast<float>(2 * i) / static_cast<float>(rotary_dim)) / scaling_factor;
        } else {
          inv_freq = 1.0f / powf(base, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        }

        float theta = static_cast<float>(pos) * inv_freq;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        // Store cos and sin values
        if (is_neox) {
          // NeoX style: [cos, cos, ..., sin, sin, ...]
          cos_sin_cache[pos * rotary_dim + i] = static_cast<T>(cos_theta);
          cos_sin_cache[pos * rotary_dim + i + rotary_dim / 2] = static_cast<T>(sin_theta);
        } else {
          // GPT-J style: [cos, sin, cos, sin, ...]
          cos_sin_cache[pos * rotary_dim + 2 * i] = static_cast<T>(cos_theta);
          cos_sin_cache[pos * rotary_dim + 2 * i + 1] = static_cast<T>(sin_theta);
        }
      }
    }
  }

  template <typename T>
  void ApplyRotaryEmbeddingOnCPU(const T* cos_sin_cache, T* query, T* key, const int64_t* positions,
                                 const int64_t* mask, const int num_tokens, const int head_size, const int num_heads,
                                 const int num_kv_heads, const int query_stride, const int key_stride,
                                 const bool is_neox) {
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
      if (mask[token_idx] == 0) continue;

      int64_t pos = positions[token_idx];

      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        T* q_ptr = query + token_idx * query_stride + head_idx * head_size;

        if (is_neox) {
          // NeoX style
          const int rotary_dim = head_size;
          for (int i = 0; i < rotary_dim / 2; ++i) {
            float q_i = static_cast<float>(q_ptr[i]);
            float q_i_plus_half = static_cast<float>(q_ptr[i + rotary_dim / 2]);

            float cos_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + i]);
            float sin_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + i + rotary_dim / 2]);

            q_ptr[i] = static_cast<T>(q_i * cos_theta - q_i_plus_half * sin_theta);
            q_ptr[i + rotary_dim / 2] = static_cast<T>(q_i * sin_theta + q_i_plus_half * cos_theta);
          }
        } else {
          const int rotary_dim = head_size;
          for (int i = 0; i < rotary_dim / 2; ++i) {
            float q_2i = static_cast<float>(q_ptr[2 * i]);
            float q_2i_plus_1 = static_cast<float>(q_ptr[2 * i + 1]);

            float cos_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + 2 * i]);
            float sin_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + 2 * i + 1]);

            q_ptr[2 * i] = static_cast<T>(q_2i * cos_theta - q_2i_plus_1 * sin_theta);
            q_ptr[2 * i + 1] = static_cast<T>(q_2i * sin_theta + q_2i_plus_1 * cos_theta);
          }
        }
      }

      for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
        T* k_ptr = key + token_idx * key_stride + head_idx * head_size;

        if (is_neox) {
          const int rotary_dim = head_size;
          for (int i = 0; i < rotary_dim / 2; ++i) {
            float k_i = static_cast<float>(k_ptr[i]);
            float k_i_plus_half = static_cast<float>(k_ptr[i + rotary_dim / 2]);

            float cos_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + i]);
            float sin_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + i + rotary_dim / 2]);

            k_ptr[i] = static_cast<T>(k_i * cos_theta - k_i_plus_half * sin_theta);
            k_ptr[i + rotary_dim / 2] = static_cast<T>(k_i * sin_theta + k_i_plus_half * cos_theta);
          }
        } else {
          const int rotary_dim = head_size;
          for (int i = 0; i < rotary_dim / 2; ++i) {
            float k_2i = static_cast<float>(k_ptr[2 * i]);
            float k_2i_plus_1 = static_cast<float>(k_ptr[2 * i + 1]);

            float cos_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + 2 * i]);
            float sin_theta = static_cast<float>(cos_sin_cache[pos * rotary_dim + 2 * i + 1]);

            k_ptr[2 * i] = static_cast<T>(k_2i * cos_theta - k_2i_plus_1 * sin_theta);
            k_ptr[2 * i + 1] = static_cast<T>(k_2i * sin_theta + k_2i_plus_1 * cos_theta);
          }
        }
      }
    }
  }

 protected:
  template <typename T>
  void ForwardLlamaRotaryEmbedding() {
    using DataType = T;
    int query_stride = num_heads * head_size;
    int key_stride = num_kv_heads * head_size;
    BufferMeta positions_cpu_meta = CreateBuffer<int64_t>(MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens)},
                                                          true, 0, max_position_embeddings);
    int cpu_rotary_pos_idx = 0;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      for (int pos = 0; pos < num_tokens; ++pos) {
        (reinterpret_cast<int64_t*>(positions_cpu_meta.data_ptr))[cpu_rotary_pos_idx++] = static_cast<int64_t>(pos);
      }
    }
    BufferMeta positions_meta = CopyToDevice<int64_t>(positions_cpu_meta);
    BufferMeta mask_cpu_meta =
        CreateBuffer<int64_t>(MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens)}, true, 1);
    BufferMeta mask_meta = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens)}, true, 1);

    BufferMeta query_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(query_stride)}, true, 0, 1);
    BufferMeta query_cpu_meta = CopyToHost<DataType>(query_meta);

    BufferMeta key_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(key_stride)}, true, 0, 1);
    BufferMeta key_cpu_meta = CopyToHost<DataType>(key_meta);

    BufferMeta cos_sin_cache_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(max_position_embeddings), static_cast<size_t>(rotary_dim)});
    BufferMeta cos_sin_cache_cpu_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_CPU, {static_cast<size_t>(max_position_embeddings), static_cast<size_t>(rotary_dim)});

    RotaryEmbeddingOnCPU<DataType>(static_cast<DataType*>(cos_sin_cache_cpu_meta.data_ptr), rotary_dim,
                                   max_position_embeddings, base, is_neox, rotary_embedding_type, scaling_factor);

    BufferMeta query_cpu_ref_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(query_stride)}, false);
    memcpy(query_cpu_ref_meta.data_ptr, query_cpu_meta.data_ptr, num_tokens * query_stride * sizeof(DataType));

    BufferMeta key_cpu_ref_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(key_stride)}, false);
    memcpy(key_cpu_ref_meta.data_ptr, key_cpu_meta.data_ptr, num_tokens * key_stride * sizeof(DataType));

    ApplyRotaryEmbeddingOnCPU<DataType>(
        static_cast<DataType*>(cos_sin_cache_cpu_meta.data_ptr), static_cast<DataType*>(query_cpu_ref_meta.data_ptr),
        static_cast<DataType*>(key_cpu_ref_meta.data_ptr), static_cast<int64_t*>(positions_cpu_meta.data_ptr),
        static_cast<int64_t*>(mask_cpu_meta.data_ptr), num_tokens, head_size, num_heads, num_kv_heads, query_stride,
        key_stride, is_neox);

    BufferMeta cos_sin_cache_ref_meta = CopyToDevice<DataType>(cos_sin_cache_cpu_meta);

    RotaryEmbeddingCuda op;
    op.SetConfig<DataType>(cos_sin_cache_meta.data_ptr, rotary_dim, max_position_embeddings, base, head_size, head_size,
                           num_heads, num_kv_heads, query_stride, is_neox, stream, rotary_embedding_type,
                           scaling_factor);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    op.SetInput(static_cast<int64_t*>(positions_meta.data_ptr), static_cast<int64_t*>(mask_meta.data_ptr),
                query_meta.data_ptr, key_meta.data_ptr, num_tokens, stream);
    op.Forward<DataType>();
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta query_ref_meta = CopyToDevice<DataType>(query_cpu_ref_meta);
    BufferMeta key_ref_meta = CopyToDevice<DataType>(key_cpu_ref_meta);

    std::string type_name = std::is_same<DataType, half>::value ? "half" : "float";
    std::string embedding_type =
        rotary_embedding_type == RotaryEmbeddingType::DEFAULT
            ? "default"
            : (rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING ? "dynamic_ntk" : "other");

    float atol = std::is_same<DataType, half>::value ? 1e-3f : 1e-5f;
    float rtol = std::is_same<DataType, half>::value ? 1e-3f : 1e-5f;

    CheckResult<DataType>("rotary_embedding_cos_sin_cache_" + type_name + "_" + embedding_type, cos_sin_cache_meta,
                          cos_sin_cache_ref_meta, atol, rtol);

    CheckResult<DataType>("rotary_embedding_query_" + type_name + "_" + embedding_type, query_meta, query_ref_meta,
                          atol, rtol);

    CheckResult<DataType>("rotary_embedding_key_" + type_name + "_" + embedding_type, key_meta, key_ref_meta, atol,
                          rtol);
  }

  template <typename T>
  void TestRotaryEmbeddingPerformance() {
    using DataType = T;
    int query_stride = num_heads * head_size;
    int key_stride = num_kv_heads * head_size;

    BufferMeta positions_cpu_meta = CreateBuffer<int64_t>(MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens)},
                                                          true, 0, max_position_embeddings);
    int cpu_rotary_pos_idx = 0;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      for (int pos = 0; pos < num_tokens; ++pos) {
        (reinterpret_cast<int64_t*>(positions_cpu_meta.data_ptr))[cpu_rotary_pos_idx++] = static_cast<int64_t>(pos);
      }
    }
    BufferMeta positions_meta = CopyToDevice<int64_t>(positions_cpu_meta);
    BufferMeta mask_meta = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens)}, true, 1);

    BufferMeta query_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(query_stride)}, true, 0, 1);

    BufferMeta key_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(key_stride)}, true, 0, 1);

    BufferMeta cos_sin_cache_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(max_position_embeddings), static_cast<size_t>(rotary_dim)});

    RotaryEmbeddingCuda op;
    op.SetConfig<DataType>(cos_sin_cache_meta.data_ptr, rotary_dim, max_position_embeddings, base, head_size, head_size,
                           num_heads, num_kv_heads, query_stride, is_neox, stream, rotary_embedding_type,
                           scaling_factor);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    op.SetInput(static_cast<int64_t*>(positions_meta.data_ptr), static_cast<int64_t*>(mask_meta.data_ptr),
                query_meta.data_ptr, key_meta.data_ptr, num_tokens, stream);

    for (int i = 0; i < 10; ++i) {
      op.Forward<DataType>();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
      op.Forward<DataType>();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    double time_ms = elapsed.count() / num_iterations;

    size_t total_elements = num_tokens * (query_stride + key_stride);
    double throughput = (total_elements * sizeof(DataType)) / (time_ms * 1e-3) / 1e9;  // GB/s

    std::cout << "RotaryEmbedding Performance (" << sizeof(DataType) << " bytes):" << std::endl;
    std::cout << "  Type: "
              << (rotary_embedding_type == RotaryEmbeddingType::DEFAULT
                      ? "DEFAULT"
                      : (rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING ? "DYNAMIC_NTK_SCALING"
                                                                                           : "OTHER"))
              << std::endl;
    std::cout << "  Num Tokens: " << num_tokens << std::endl;
    std::cout << "  Num Heads: " << num_heads << std::endl;
    std::cout << "  Head Size: " << head_size << std::endl;
    std::cout << "  Rotary Dim: " << rotary_dim << std::endl;
    std::cout << "  Average Time: " << time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
  }
};

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaRotaryEmbeddingHalfTest) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  ForwardLlamaRotaryEmbedding<half>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaRotaryEmbeddingFloatTest) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  ForwardLlamaRotaryEmbedding<float>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaRotaryEmbeddingBFloat16Test) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  ForwardLlamaRotaryEmbedding<__nv_bfloat16>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaDynamicNTKScalingRotaryEmbeddingHalfTest) {
  rotary_embedding_type = RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
  scaling_factor = 4.0f;
  ForwardLlamaRotaryEmbedding<half>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaDynamicNTKScalingRotaryEmbeddingFloatTest) {
  rotary_embedding_type = RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
  scaling_factor = 4.0f;
  ForwardLlamaRotaryEmbedding<float>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaDynamicNTKScalingRotaryEmbeddingBFloat16Test) {
  rotary_embedding_type = RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
  scaling_factor = 4.0f;
  ForwardLlamaRotaryEmbedding<__nv_bfloat16>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaLinearScalingRotaryEmbeddingHalfTest) {
  rotary_embedding_type = RotaryEmbeddingType::LINEAR_SCALING;
  scaling_factor = 2.0f;
  ForwardLlamaRotaryEmbedding<half>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaLinearScalingRotaryEmbeddingFloatTest) {
  rotary_embedding_type = RotaryEmbeddingType::LINEAR_SCALING;
  scaling_factor = 2.0f;
  ForwardLlamaRotaryEmbedding<float>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaLinearScalingRotaryEmbeddingBFloat16Test) {
  rotary_embedding_type = RotaryEmbeddingType::LINEAR_SCALING;
  scaling_factor = 2.0f;
  ForwardLlamaRotaryEmbedding<__nv_bfloat16>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, RotaryEmbeddingPerformanceHalfTest) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  TestRotaryEmbeddingPerformance<half>();

  num_tokens = 1024;
  TestRotaryEmbeddingPerformance<half>();

  num_tokens = 2048;
  TestRotaryEmbeddingPerformance<half>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, RotaryEmbeddingPerformanceFloatTest) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  TestRotaryEmbeddingPerformance<float>();

  num_tokens = 1024;
  TestRotaryEmbeddingPerformance<float>();

  num_tokens = 2048;
  TestRotaryEmbeddingPerformance<float>();
}

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, RotaryEmbeddingPerformanceBFloat16Test) {
  rotary_embedding_type = RotaryEmbeddingType::DEFAULT;
  scaling_factor = 1.0f;
  TestRotaryEmbeddingPerformance<__nv_bfloat16>();

  num_tokens = 1024;
  TestRotaryEmbeddingPerformance<__nv_bfloat16>();

  num_tokens = 2048;
  TestRotaryEmbeddingPerformance<__nv_bfloat16>();
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
