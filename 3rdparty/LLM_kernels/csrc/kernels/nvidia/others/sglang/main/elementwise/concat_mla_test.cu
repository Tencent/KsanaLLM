/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/others/sglang/main/elementwise/concat_mla.h"

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaConcatMlaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // Config of DeepSeek
  const int qk_nope_head_dim = 128;
  const int qk_rope_head_dim = 64;
  const int k_head_dim = qk_nope_head_dim + qk_rope_head_dim;

  template <typename T>
  void TestConcatMlaK(const int num_tokens, const int num_heads, bool perf = false) {
    // Allocate device memory
    BufferMeta k_nope_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens * num_heads * qk_nope_head_dim)},
                        /*is_random_init*/ true);
    BufferMeta k_rope_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens * qk_rope_head_dim)},
                        /*is_random_init*/ true);
    BufferMeta k_rope_expand_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens * num_heads * qk_rope_head_dim)});
    BufferMeta k_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens * num_heads * k_head_dim)});
    BufferMeta k_ref_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens * num_heads * k_head_dim)});

    const int warmups = perf ? 5 : 0;
    const int iterations = perf ? 10 : 1;

    // Invoke concat_mla_k
    auto cuda_run = [&]() {
      concat_mla_k<T>(reinterpret_cast<const T*>(k_nope_meta.data_ptr),
                      reinterpret_cast<const T*>(k_rope_meta.data_ptr), reinterpret_cast<T*>(k_meta.data_ptr),
                      num_tokens, num_heads, qk_nope_head_dim, qk_rope_head_dim, stream);
    };
    // Invoke expand and concat for reference
    auto cuda_ref_run = [&]() {
      InvokeExpand<T>(reinterpret_cast<const T*>(k_rope_meta.data_ptr),
                      reinterpret_cast<T*>(k_rope_expand_meta.data_ptr), num_tokens, num_heads, qk_rope_head_dim, 0,
                      stream);
      Concat<T>(reinterpret_cast<const T*>(k_nope_meta.data_ptr),
                reinterpret_cast<const T*>(k_rope_expand_meta.data_ptr), qk_nope_head_dim, qk_rope_head_dim,
                num_tokens * num_heads, 1, reinterpret_cast<T*>(k_ref_meta.data_ptr), stream);
    };

    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    const float elapsed_ref_ms = MeasureCudaExecutionTime(cuda_ref_run, stream, warmups, iterations);
    if (perf) {
      std::cout << "Token num: " << num_tokens << ", Head num: " << num_heads << ", Execution time: " << elapsed_ms
                << " ms vs " << elapsed_ref_ms << " ms" << std::endl;
    } else {
      // Verify accuracy
      EXPECT_TRUE(CheckResult<T>(
          "concat_mla_k_num_tokens_" + std::to_string(num_tokens) + "_num_heads_" + std::to_string(num_heads), k_meta,
          k_ref_meta, 1e-4, 1e-4));
    }

    // Free data
    DeleteBuffer(k_nope_meta);
    DeleteBuffer(k_rope_meta);
    DeleteBuffer(k_rope_expand_meta);
    DeleteBuffer(k_meta);
    DeleteBuffer(k_ref_meta);
  }
};

TEST_F(LlamaNvidiaConcatMlaTestSuit, ConcatMlaKAccTest) {
  for (const int num_tokens : {7, 33}) {
    for (const int num_heads : {16, 128}) {
      TestConcatMlaK<half>(num_tokens, num_heads);
      TestConcatMlaK<__nv_bfloat16>(num_tokens, num_heads);
    }
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaConcatMlaTestSuit, DISABLED_ConcatMlaKPerfTest) {
  for (const int num_heads : {16, 128}) {
    for (const int num_tokens : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) {
      TestConcatMlaK<__nv_bfloat16>(num_tokens, num_heads, /*perf*/ true);
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
