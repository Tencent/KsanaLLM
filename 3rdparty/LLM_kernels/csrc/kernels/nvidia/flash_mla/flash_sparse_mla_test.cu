/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/flash_mla/flash_sparse_mla.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "params.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaFlashSparseMlaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    skip_test = GetSMVersion() < 90;

    NvidiaTestSuitBase::SetUp();
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  void PrepareData(const size_t batch_size, const size_t seqlen_q_ori, const size_t seqlen_k) {
    // Prepare meta data
    decoding_attn_impl_meta = GetAttnImplMeta(seqlen_q_ori * num_heads_q, num_heads_k, num_heads_q, is_fp8,
                                              /*is_sparse_attn*/ true);
    d_tile_scheduler_metadata = CreateBuffer<int>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(decoding_attn_impl_meta.num_sm_parts), TileSchedulerMetaDataSize});
    d_num_splits = CreateBuffer<int>(MemoryType::MEMORY_GPU, {batch_size + 1});
    InvokeGetSparseMlaMetadata(
        /*seqlens_k_ptr*/ nullptr, batch_size, seqlen_q_ori * num_heads_q, num_heads_k, num_heads_q, is_fp8, topk,
        stream, reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
        reinterpret_cast<int*>(d_num_splits.data_ptr));
    // Prepare indices
    h_indices = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size * seqlen_q_ori, topk});
    int* current_h_indices_ptr = reinterpret_cast<int*>(h_indices.data_ptr);
    num_blocks = 0;
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < seqlen_q_ori; j++) {
        for (size_t k = 0; k < topk; k++) {
          if (k <= seqlen_k - seqlen_q_ori + j) {
            *current_h_indices_ptr++ = num_blocks * block_size + k;
          } else {
            *current_h_indices_ptr++ = -1;
          }
        }
      }
      num_blocks += (seqlen_k + block_size - 1) / block_size;
    }
    d_indices = CopyToDevice<int>(h_indices);
    // Prepare qkv
    d_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q_ori, num_heads_q, head_size_k},
                          /*is_random_init*/ true);
    // Control the numerical values during computation
    d_kcache = CreateBuffer<char>(MemoryType::MEMORY_GPU, {num_blocks, k_batch_stride_large},
                                  /*is_random_init*/ true, /*min_val*/ 0, /*max_val*/ 1 << 5);
    // Prepare large enough workspace
    d_workspace = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1 << 28});
    // Prepare output
    d_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q_ori, num_heads_q, head_size_v});
  }

  void FreeData() {
    DeleteBuffer(d_tile_scheduler_metadata);
    DeleteBuffer(d_num_splits);
    DeleteBuffer(h_indices);
    DeleteBuffer(d_indices);
    DeleteBuffer(d_q);
    DeleteBuffer(d_kcache);
    DeleteBuffer(d_workspace);
    DeleteBuffer(d_out);
  }

  using NvidiaTestSuitBase::stream;
  using T = __nv_bfloat16;

  bool skip_test;

  // Config of DeepSeek-V32
  const size_t num_heads_q = 128;
  const size_t num_heads_k = 1;
  const size_t head_size_k = 576;
  const size_t head_size_v = 512;
  const size_t block_size = 64;
  const size_t k_batch_stride_large =
      block_size * (656 + 132);  // dim of fp8_ds_mla is 656, extended by 132 to simulate mixed cache block with indexer
  const float softmax_scale = 0.1147213867929261;
  const bool is_causal = true;
  const bool is_fp8 = true;
  const size_t topk = 2048;

  size_t num_blocks = 0;

  DecodingAttnImplMeta decoding_attn_impl_meta;

  // Data buffers for test
  BufferMeta d_tile_scheduler_metadata;
  BufferMeta d_num_splits;
  BufferMeta h_indices;
  BufferMeta d_indices;
  BufferMeta d_q;
  BufferMeta d_kcache;
  BufferMeta d_workspace;
  BufferMeta d_out;
};

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, FlashSparseMlaKernelAccTest) {
  if (skip_test) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
  }

  const size_t batch_size = 2;
  const size_t seqlen_q_ori = 2;
  const size_t seqlen_k = 100;
  const size_t max_num_blocks_per_seq = (seqlen_k + block_size - 1) / block_size;

  PrepareData(batch_size, seqlen_q_ori, seqlen_k);

  // Invoke kernel
  InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
      reinterpret_cast<T*>(d_q.data_ptr), reinterpret_cast<uint8_t*>(d_kcache.data_ptr), batch_size, seqlen_q_ori,
      num_heads_q, num_heads_k, head_size_k, head_size_v, k_batch_stride_large, max_num_blocks_per_seq, num_blocks,
      block_size, decoding_attn_impl_meta.num_sm_parts, /*seqlens_k_ptr*/ nullptr,
      /*block_table_ptr*/ nullptr, softmax_scale, is_causal, reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
      reinterpret_cast<int*>(d_num_splits.data_ptr), is_fp8, reinterpret_cast<int*>(d_indices.data_ptr), topk, stream,
      reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  auto IsBf16NaN = [](const T x) -> bool {
    uint16_t u16 = 0;
    static_assert(sizeof(uint16_t) == sizeof(T));
    std::memcpy(&u16, &x, sizeof(u16));
    constexpr uint16_t kExpMask = 0x7F80u;   // bits 14..7
    constexpr uint16_t kMantMask = 0x007Fu;  // bits 6..0
    uint16_t exp = u16 & kExpMask;
    uint16_t mant = u16 & kMantMask;
    // exp all ones and mantissa non-zero -> NaN
    return (exp == kExpMask) && (mant != 0);
  };

  // Check no NaNs in the output
  BufferMeta h_out = CopyToHost<T>(d_out);
  for (size_t i = 0; i < batch_size * seqlen_q_ori * num_heads_q * head_size_v; i++) {
    EXPECT_FALSE(IsBf16NaN(reinterpret_cast<T*>(h_out.data_ptr)[i]));
  }

  FreeData();
  DeleteBuffer(h_out);
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelPerfTest) {
  if (skip_test) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
  }

  const size_t batch_size = 1;

  for (const size_t seqlen_k : {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) {
    const size_t seqlen_q_ori = 1;
    const size_t max_num_blocks_per_seq = (seqlen_k + block_size - 1) / block_size;
    PrepareData(batch_size, seqlen_q_ori, seqlen_k);

    // Performance run
    const int warmups = 5;
    const int iterations = 10;
    auto cuda_run = [&]() {
      InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
          reinterpret_cast<T*>(d_q.data_ptr), reinterpret_cast<uint8_t*>(d_kcache.data_ptr), batch_size, seqlen_q_ori,
          num_heads_q, num_heads_k, head_size_k, head_size_v, k_batch_stride_large, max_num_blocks_per_seq, num_blocks,
          block_size, decoding_attn_impl_meta.num_sm_parts, /*seqlens_k_ptr*/ nullptr,
          /*block_table_ptr*/ nullptr, softmax_scale, is_causal,
          reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr), reinterpret_cast<int*>(d_num_splits.data_ptr),
          is_fp8, reinterpret_cast<int*>(d_indices.data_ptr), topk, stream,
          reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr));
    };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Seq len k: " << seqlen_k << ", Execution time of flash sparse mla bf16: " << elapsed_ms << " ms"
              << std::endl;

    FreeData();
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaPrefillKernelPerfTest) {
  if (skip_test) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
  }

  const size_t batch_size = 1;

  for (const size_t seqlen_k : {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) {
    const size_t seqlen_q = seqlen_k;
    // Prepare indices
    BufferMeta h_indices = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size * seqlen_q, topk});
    int* current_h_indices_ptr = reinterpret_cast<int*>(h_indices.data_ptr);
    int prefix_len = 0;
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < seqlen_q; j++) {
        for (size_t k = 0; k < topk; k++) {
          if (k <= seqlen_k - seqlen_q + j) {
            *current_h_indices_ptr++ = prefix_len + k;
          } else {
            *current_h_indices_ptr++ = -1;
          }
        }
      }
      prefix_len += seqlen_k;
    }
    d_indices = CopyToDevice<int>(h_indices);
    DeleteBuffer(h_indices);
    // Prepare qkv
    d_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, head_size_k},
                          /*is_random_init*/ false);
    // Control the numerical values during computation
    d_kcache = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_k, num_heads_k, head_size_k},
                               /*is_random_init*/ true, /*min_val*/ 0, /*max_val*/ 1 << 5);
    // Prepare large enough workspace
    d_workspace = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1 << 28});
    // Prepare output
    d_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, head_size_v});

    // Performance run
    const int warmups = 5;
    const int iterations = 10;
    auto cuda_run = [&]() {
      InvokeFlashSparseMlaPrefill<T, T, KVCacheType::kAuto>(
          reinterpret_cast<T*>(d_q.data_ptr), reinterpret_cast<T*>(d_kcache.data_ptr),
          reinterpret_cast<int*>(d_indices.data_ptr), seqlen_q, seqlen_k, num_heads_q, num_heads_k, head_size_k,
          head_size_v, topk, softmax_scale, stream, reinterpret_cast<float*>(d_workspace.data_ptr),
          reinterpret_cast<T*>(d_out.data_ptr));
    };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Seq len k: " << seqlen_k << ", Execution time of flash sparse mla prefill bf16: " << elapsed_ms
              << " ms" << std::endl;

    DeleteBuffer(d_indices);
    DeleteBuffer(d_q);
    DeleteBuffer(d_kcache);
    DeleteBuffer(d_workspace);
    DeleteBuffer(d_out);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
