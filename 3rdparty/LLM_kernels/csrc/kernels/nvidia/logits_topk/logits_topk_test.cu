/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/logits_topk/logits_topk.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

namespace {

constexpr int32_t kTopK = 2048;

template <typename T>
std::vector<T> CopyDeviceVector(const BufferMeta& buffer) {
  std::vector<T> host(buffer.n_elmts);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(host.data(), buffer.data_ptr, buffer.buf_size, cudaMemcpyDeviceToHost));
  return host;
}

}  // namespace

class NvidiaLogitsTopKTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  void RunShortRowTest() {
    const int64_t num_rows = 2;
    const int32_t row_lengths[num_rows] = {5, 10};
    const int32_t stride0 = 16;
    const int32_t stride1 = 1;

    BufferMeta logits_buffer = CreateBuffer<float>(MEMORY_GPU, {static_cast<size_t>(num_rows), static_cast<size_t>(stride0)});
    BufferMeta row_starts_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows)});
    BufferMeta row_ends_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows)});
    BufferMeta indices_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows), static_cast<size_t>(kTopK)});

    std::vector<float> logits_host(num_rows * stride0, -1e6f);
    std::vector<int32_t> row_starts(num_rows, 0);
    std::vector<int32_t> row_ends(num_rows);

    for (int64_t row = 0; row < num_rows; ++row) {
      row_ends[row] = row_lengths[row];
      for (int32_t col = 0; col < row_lengths[row]; ++col) {
        logits_host[row * stride0 + col * stride1] = row * 10.0f + static_cast<float>(col);
      }
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(logits_buffer.data_ptr, logits_host.data(),
                                       sizeof(float) * logits_host.size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_starts_buffer.data_ptr, row_starts.data(),
                                       sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_ends_buffer.data_ptr, row_ends.data(),
                                       sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(indices_buffer.data_ptr, 0, sizeof(int32_t) * num_rows * kTopK));

    InvokeFastTopK(reinterpret_cast<const float*>(logits_buffer.data_ptr),
                   reinterpret_cast<const int32_t*>(row_starts_buffer.data_ptr),
                   reinterpret_cast<const int32_t*>(row_ends_buffer.data_ptr),
                   reinterpret_cast<int32_t*>(indices_buffer.data_ptr),
                   num_rows,
                   stride0,
                   stride1,
                   stream);

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto indices_host = CopyDeviceVector<int32_t>(indices_buffer);

    for (int64_t row = 0; row < num_rows; ++row) {
      const int32_t len = row_lengths[row];
      for (int32_t i = 0; i < len; ++i) {
        EXPECT_EQ(indices_host[row * kTopK + i], i) << "Unexpected index for row " << row << " at position " << i;
      }
      for (int32_t i = len; i < kTopK; ++i) {
        EXPECT_EQ(indices_host[row * kTopK + i], -1) << "Expected padding index -1 for row " << row << " at position " << i;
      }
    }

    DeleteBuffer(indices_buffer);
    DeleteBuffer(row_ends_buffer);
    DeleteBuffer(row_starts_buffer);
    DeleteBuffer(logits_buffer);
  }

  void RunLargeRowTest(int32_t row_start) {
    const int64_t num_rows = 1;
    const int32_t row_len = 2300;
    const int32_t stride0 = row_start + row_len + 16;
    const int32_t stride1 = 1;
    const int32_t row_end = row_start + row_len;

    BufferMeta logits_buffer = CreateBuffer<float>(MEMORY_GPU, {static_cast<size_t>(num_rows), static_cast<size_t>(stride0)});
    BufferMeta row_starts_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows)});
    BufferMeta row_ends_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows)});
    BufferMeta indices_buffer = CreateBuffer<int32_t>(MEMORY_GPU, {static_cast<size_t>(num_rows), static_cast<size_t>(kTopK)});

    std::vector<float> logits_host(stride0, -1e6f);
    std::vector<int32_t> row_starts(num_rows, row_start);
    std::vector<int32_t> row_ends(num_rows, row_end);

    for (int32_t idx = row_start; idx < row_end; ++idx) {
      logits_host[idx * stride1] = static_cast<float>(idx);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(logits_buffer.data_ptr, logits_host.data(),
                                       sizeof(float) * logits_host.size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_starts_buffer.data_ptr, row_starts.data(),
                                       sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(row_ends_buffer.data_ptr, row_ends.data(),
                                       sizeof(int32_t) * num_rows, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(indices_buffer.data_ptr, 0, sizeof(int32_t) * num_rows * kTopK));

    InvokeFastTopK(reinterpret_cast<const float*>(logits_buffer.data_ptr),
                   reinterpret_cast<const int32_t*>(row_starts_buffer.data_ptr),
                   reinterpret_cast<const int32_t*>(row_ends_buffer.data_ptr),
                   reinterpret_cast<int32_t*>(indices_buffer.data_ptr),
                   num_rows,
                   stride0,
                   stride1,
                   stream);

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto indices_host = CopyDeviceVector<int32_t>(indices_buffer);

    std::vector<std::pair<float, int32_t>> candidates(row_len);
    for (int32_t i = 0; i < row_len; ++i) {
      const float val = logits_host[(row_start + i) * stride1];
      candidates[i] = {val, i};
    }

    const int32_t top_count = std::min(kTopK, row_len);
    std::partial_sort(candidates.begin(), candidates.begin() + top_count, candidates.end(),
                      [](const auto& lhs, const auto& rhs) {
                        if (lhs.first == rhs.first) {
                          return lhs.second < rhs.second;
                        }
                        return lhs.first > rhs.first;
                      });

    for (int32_t i = 0; i < top_count; ++i) {
      EXPECT_EQ(indices_host[i], candidates[i].second)
          << "Mismatch at position " << i << " expected index " << candidates[i].second
          << " but got " << indices_host[i];
    }

    DeleteBuffer(indices_buffer);
    DeleteBuffer(row_ends_buffer);
    DeleteBuffer(row_starts_buffer);
    DeleteBuffer(logits_buffer);
  }
};

TEST_F(NvidiaLogitsTopKTestSuit, HandlesRowsShorterThanTopK) { RunShortRowTest(); }

TEST_F(NvidiaLogitsTopKTestSuit, ExtractsTopKDescendingWithZeroStart) { RunLargeRowTest(/*row_start=*/0); }

TEST_F(NvidiaLogitsTopKTestSuit, ExtractsTopKDescendingWithOffsetStart) { RunLargeRowTest(/*row_start=*/64); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
