/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "concat.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaConcatTestSuit : public NvidiaTestSuitBase {
 public:
  LlamaNvidiaConcatTestSuit(): random_range(10, 200) {}
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  std::default_random_engine random_engine;
  std::uniform_int_distribution<size_t> random_range;
};

TEST_F(LlamaNvidiaConcatTestSuit, ConcatFirstDimTest) {
  using data_type = half;
  for (int i = 0; i < 5; ++i) {
    const size_t kDim1 = random_range(random_engine);
    const size_t kDim2 = random_range(random_engine);
    const size_t kConcatDimA = random_range(random_engine);
    const size_t kConcatDimB = random_range(random_engine);

    BufferMeta input_a = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kConcatDimA * kDim1 * kDim2}, true);
    BufferMeta input_b = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kConcatDimB * kDim1 * kDim2}, true);
    BufferMeta output = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {(kConcatDimA + kConcatDimB) * kDim1 * kDim2});

    BufferMeta input_a_host = CopyToHost<data_type>(input_a);
    BufferMeta input_b_host = CopyToHost<data_type>(input_b);

    Concat<data_type>(reinterpret_cast<const data_type*>(input_a.data_ptr),
                      reinterpret_cast<const data_type*>(input_b.data_ptr), kConcatDimA, kConcatDimB, 1, kDim1 * kDim2,
                      reinterpret_cast<data_type*>(output.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta output_host = CopyToHost<data_type>(output);

    const data_type* output_ptr = reinterpret_cast<data_type*>(output_host.data_ptr);
    const data_type* input_a_ptr = reinterpret_cast<data_type*>(input_a_host.data_ptr);
    const data_type* input_b_ptr = reinterpret_cast<data_type*>(input_b_host.data_ptr);
    for (size_t x = 0; x < kConcatDimA; ++x) {
      for (size_t y = 0; y < kDim1; ++y) {
        for (size_t z = 0; z < kDim2; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_a_ptr++));
        }
      }
    }
    for (size_t x = 0; x < kConcatDimB; ++x) {
      for (size_t y = 0; y < kDim1; ++y) {
        for (size_t z = 0; z < kDim2; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_b_ptr++));
        }
      }
    }
  }
}

TEST_F(LlamaNvidiaConcatTestSuit, ConcatMidDimTest) {
  using data_type = half;
  for (int i = 0; i < 5; ++i) {
    const size_t kDim0 = random_range(random_engine);
    const size_t kDim2 = random_range(random_engine);
    const size_t kConcatDimA = random_range(random_engine);
    const size_t kConcatDimB = random_range(random_engine);

    BufferMeta input_a = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * kConcatDimA * kDim2}, true);
    BufferMeta input_b = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * kConcatDimB * kDim2}, true);
    BufferMeta output = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * (kConcatDimA + kConcatDimB) * kDim2});

    BufferMeta input_a_host = CopyToHost<data_type>(input_a);
    BufferMeta input_b_host = CopyToHost<data_type>(input_b);

    Concat<data_type>(reinterpret_cast<const data_type*>(input_a.data_ptr),
                      reinterpret_cast<const data_type*>(input_b.data_ptr), kConcatDimA, kConcatDimB, kDim0, kDim2,
                      reinterpret_cast<data_type*>(output.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta output_host = CopyToHost<data_type>(output);

    const data_type* output_ptr = reinterpret_cast<data_type*>(output_host.data_ptr);
    const data_type* input_a_ptr = reinterpret_cast<data_type*>(input_a_host.data_ptr);
    const data_type* input_b_ptr = reinterpret_cast<data_type*>(input_b_host.data_ptr);
    for (size_t x = 0; x < kDim0; ++x) {
      for (size_t y = 0; y < kConcatDimA; ++y) {
        for (size_t z = 0; z < kDim2; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_a_ptr++));
        }
      }
      for (size_t y = 0; y < kConcatDimB; ++y) {
        for (size_t z = 0; z < kDim2; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_b_ptr++));
        }
      }
    }
  }
}

TEST_F(LlamaNvidiaConcatTestSuit, ConcatLastDimTest) {
  using data_type = half;
  for (int i = 0; i < 5; ++i) {
    const size_t kDim0 = random_range(random_engine);
    const size_t kDim1 = random_range(random_engine);
    const size_t kConcatDimA = random_range(random_engine);
    const size_t kConcatDimB = random_range(random_engine);


    BufferMeta input_a = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * kDim1 * kConcatDimA}, true);
    BufferMeta input_b = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * kDim1 * kConcatDimB}, true);
    BufferMeta output = CreateBuffer<data_type>(MemoryType::MEMORY_GPU, {kDim0 * kDim1 * (kConcatDimA + kConcatDimB)});

    BufferMeta input_a_host = CopyToHost<data_type>(input_a);
    BufferMeta input_b_host = CopyToHost<data_type>(input_b);

    Concat<data_type>(reinterpret_cast<const data_type*>(input_a.data_ptr),
                      reinterpret_cast<const data_type*>(input_b.data_ptr), kConcatDimA, kConcatDimB, kDim0 * kDim1, 1,
                      reinterpret_cast<data_type*>(output.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta output_host = CopyToHost<data_type>(output);

    const data_type* output_ptr = reinterpret_cast<data_type*>(output_host.data_ptr);
    const data_type* input_a_ptr = reinterpret_cast<data_type*>(input_a_host.data_ptr);
    const data_type* input_b_ptr = reinterpret_cast<data_type*>(input_b_host.data_ptr);
    for (size_t x = 0; x < kDim0; ++x) {
      for (size_t y = 0; y < kDim1; ++y) {
        for (size_t z = 0; z < kConcatDimA; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_a_ptr++));
        }
        for (size_t z = 0; z < kConcatDimB; ++z) {
          EXPECT_FLOAT_EQ(static_cast<float>(*output_ptr++), static_cast<float>(*input_b_ptr++));
        }
      }
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
