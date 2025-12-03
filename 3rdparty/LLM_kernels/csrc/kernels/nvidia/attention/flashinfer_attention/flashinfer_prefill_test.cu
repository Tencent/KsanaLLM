/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/attention/flashinfer_attention/flashinfer_prefill.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

// Function to validate attention output
template <typename T>
void ValidateAttentionOutput(const BufferMeta& output, const std::string& test_name, cudaStream_t stream) {
  size_t output_size = output.n_elmts;
  std::vector<T> h_output(output_size);

  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(h_output.data(), output.data_ptr, sizeof(T) * output_size, cudaMemcpyDeviceToHost));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  size_t zero_count = 0;
  size_t nan_count = 0;
  size_t inf_count = 0;
  float sum = 0.0f;
  float max_val = -std::numeric_limits<float>::infinity();
  float min_val = std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < output_size; ++i) {
    float val = static_cast<float>(h_output[i]);

    if (std::isnan(val)) {
      nan_count++;
    } else if (std::isinf(val)) {
      inf_count++;
    } else if (val == 0.0f) {
      zero_count++;
    } else {
      sum += val;
      max_val = std::max(max_val, val);
      min_val = std::min(min_val, val);
    }
  }

  EXPECT_EQ(nan_count, 0) << test_name << ": Output contains NaN values";
  EXPECT_EQ(inf_count, 0) << test_name << ": Output contains Inf values";
  EXPECT_LT(zero_count, output_size) << test_name << ": Output is all zeros";
}

class FlashinferPrefillTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  // Test configuration parameters
  int num_seqs{2};
  int num_heads{32};
  int num_kv_heads{4};
  int head_size{128};
  int block_size{16};
  int max_context_len{256};
  float softmax_scale{1.0f / sqrt(128.0f)};
  bool is_causal{true};
  size_t pinned_workspace_size{8 * 1024 * 1024};
};

TEST_F(FlashinferPrefillTestSuit, FlashinferPrefillHalfTest) {
#if !defined(ENABLE_FLASHINFER)
  GTEST_SKIP() << "ENABLE_FLASHINFER is not defined. skipping FlashInfer test." << std::endl;
#endif
  size_t max_blocks_per_seq = (max_context_len + block_size - 1) / block_size;
  using DataType = half;
  // Create query buffer: [num_seqs, num_heads, head_size]
  BufferMeta query = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}, true, 0, 1);

  // Create output buffer: [num_seqs, num_heads, head_size]
  BufferMeta out = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});

  // Create context_lens buffer: [num_seqs]
  BufferMeta context_lens = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs)});
  // Fill context_lens with max_context_len
  InvokeRange(static_cast<int*>(context_lens.data_ptr), max_context_len, num_seqs, 0, stream);

  // Create KV cache buffers
  size_t total_blocks = num_seqs * max_blocks_per_seq;
  BufferMeta k_cache = CreateBuffer<DataType>(MemoryType::MEMORY_GPU,
                                              {total_blocks, static_cast<size_t>(block_size),
                                               static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                              true, 0, 1);
  BufferMeta v_cache = CreateBuffer<DataType>(MemoryType::MEMORY_GPU,
                                              {total_blocks, static_cast<size_t>(block_size),
                                               static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                              true, 0, 1);

  // Create block_table buffer: [num_seqs, max_blocks_per_seq]
  BufferMeta block_table = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(total_blocks)});
  // Initialize block_table with sequential indices: 0, 1, 2, ..., total_blocks-1
  InvokeRange(static_cast<int*>(block_table.data_ptr), 0, total_blocks, 1, stream);

  // Allocate workspace_buffer
  size_t workspace_size = GetFlashInferDeviceWorkspaceSize(num_heads, num_kv_heads, head_size);
  BufferMeta workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size});

  size_t flashinfer_extra_workspace_size = workspace_size;
  BufferMeta flashinfer_extra_workspace_buffer =
      CreateBuffer<char>(MemoryType::MEMORY_GPU, {flashinfer_extra_workspace_size});

  // Page-locked workspace_buffer for FlashInfer helper
  BufferMeta pinned_workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_CPU_PINNED, {pinned_workspace_size});

  // Create FlashinferBatchPrefillHelper
  FlashinferBatchPrefillHelper<DataType, llm_kernels::utils::KVCacheType::kAuto, DataType, int32_t> helper;

  // Prepare
  helper.Prepare(num_heads, head_size, num_kv_heads, block_size, max_blocks_per_seq, num_seqs, query.data_ptr,
                 out.data_ptr, context_lens.data_ptr, k_cache.data_ptr, v_cache.data_ptr,
                 reinterpret_cast<int32_t*>(block_table.data_ptr), nullptr, is_causal, softmax_scale,
                 workspace_buffer.data_ptr, workspace_size, flashinfer_extra_workspace_buffer.data_ptr,
                 pinned_workspace_buffer.data_ptr, true, stream);

  // Forward
  helper.Forward();

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Validate output
  ValidateAttentionOutput<DataType>(out, "FlashinferPrefillHalfTest", stream);

  DeleteBuffer(query);
  DeleteBuffer(out);
  DeleteBuffer(k_cache);
  DeleteBuffer(v_cache);
  DeleteBuffer(block_table);
  DeleteBuffer(context_lens);
  DeleteBuffer(workspace_buffer);
  DeleteBuffer(flashinfer_extra_workspace_buffer);
  DeleteBuffer(pinned_workspace_buffer);
}

TEST_F(FlashinferPrefillTestSuit, FlashinferPrefillBFloat16Test) {
#if !defined(ENABLE_FLASHINFER)
  GTEST_SKIP() << "ENABLE_FLASHINFER is not defined. skipping FlashInfer test." << std::endl;
#endif
  size_t max_blocks_per_seq = (max_context_len + block_size - 1) / block_size;
  using DataType = __nv_bfloat16;
  // Create query buffer
  BufferMeta query = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}, true, 0, 1);

  // Create output buffer
  BufferMeta out = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});

  // Create context_lens buffer
  BufferMeta context_lens = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs)});
  InvokeRange(static_cast<int*>(context_lens.data_ptr), max_context_len, num_seqs, 0, stream);

  // Create KV cache buffers
  size_t total_blocks = num_seqs * max_blocks_per_seq;
  BufferMeta k_cache = CreateBuffer<DataType>(MemoryType::MEMORY_GPU,
                                              {total_blocks, static_cast<size_t>(block_size),
                                               static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                              true, 0, 1);
  BufferMeta v_cache = CreateBuffer<DataType>(MemoryType::MEMORY_GPU,
                                              {total_blocks, static_cast<size_t>(block_size),
                                               static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                              true, 0, 1);

  // Create block_table buffer
  BufferMeta block_table = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(total_blocks)});
  InvokeRange(static_cast<int*>(block_table.data_ptr), 0, total_blocks, 1, stream);

  // Allocate workspace_buffer
  size_t workspace_size = GetFlashInferDeviceWorkspaceSize(num_heads, num_kv_heads, head_size);
  BufferMeta workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size});
  BufferMeta flashinfer_extra_workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size});
  BufferMeta pinned_workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_CPU_PINNED, {pinned_workspace_size});

  // Create FlashinferBatchPrefillHelper
  FlashinferBatchPrefillHelper<DataType, llm_kernels::utils::KVCacheType::kAuto, DataType, int32_t> helper;

  // Prepare
  helper.Prepare(num_heads, head_size, num_kv_heads, block_size, max_blocks_per_seq, num_seqs, query.data_ptr,
                 out.data_ptr, context_lens.data_ptr, k_cache.data_ptr, v_cache.data_ptr,
                 reinterpret_cast<int32_t*>(block_table.data_ptr), nullptr, is_causal, softmax_scale,
                 workspace_buffer.data_ptr, workspace_size, flashinfer_extra_workspace_buffer.data_ptr,
                 pinned_workspace_buffer.data_ptr, true, stream);

  // Forward
  helper.Forward();

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Validate output
  ValidateAttentionOutput<DataType>(out, "FlashinferPrefillBFloat16Test", stream);

  DeleteBuffer(query);
  DeleteBuffer(out);
  DeleteBuffer(k_cache);
  DeleteBuffer(v_cache);
  DeleteBuffer(block_table);
  DeleteBuffer(context_lens);
  DeleteBuffer(workspace_buffer);
  DeleteBuffer(flashinfer_extra_workspace_buffer);
  DeleteBuffer(pinned_workspace_buffer);
}

#ifdef ENABLE_FP8
TEST_F(FlashinferPrefillTestSuit, FlashinferPrefillFP8E4M3Test) {
#  if !defined(ENABLE_FLASHINFER)
  GTEST_SKIP() << "ENABLE_FLASHINFER is not defined. skipping FlashInfer test." << std::endl;
#  endif
  size_t max_blocks_per_seq = (max_context_len + block_size - 1) / block_size;
  // Create buffers
  BufferMeta query_half = CreateBuffer<half>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}, true, 0, 1);

  BufferMeta out_half = CreateBuffer<half>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});

  BufferMeta query_bfloat16 = CreateBuffer<__nv_bfloat16>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}, true, 0, 1);

  BufferMeta out_bfloat16 = CreateBuffer<__nv_bfloat16>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});

  BufferMeta context_lens = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs)});
  InvokeRange(static_cast<int*>(context_lens.data_ptr), max_context_len, num_seqs, 0, stream);

  size_t total_blocks = num_seqs * max_blocks_per_seq;
  BufferMeta k_cache = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU,
                                                   {total_blocks, static_cast<size_t>(block_size),
                                                    static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                                   true, 0, 1);
  BufferMeta v_cache = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU,
                                                   {total_blocks, static_cast<size_t>(block_size),
                                                    static_cast<size_t>(num_kv_heads), static_cast<size_t>(head_size)},
                                                   true, 0, 1);

  BufferMeta block_table = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(total_blocks)});

  InvokeRange(static_cast<int*>(block_table.data_ptr), 0, total_blocks, 1, stream);

  size_t workspace_size = GetFlashInferDeviceWorkspaceSize(num_heads, num_kv_heads, head_size);
  BufferMeta workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size});
  BufferMeta flashinfer_extra_workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size});
  BufferMeta pinned_workspace_buffer = CreateBuffer<char>(MemoryType::MEMORY_CPU_PINNED, {pinned_workspace_size});

  // Test with fp8_e4m3 KV cache type and  half scalar type
  FlashinferBatchPrefillHelper<half, llm_kernels::utils::KVCacheType::kFp8E4M3, half, int32_t> helper_half;

  helper_half.Prepare(num_heads, head_size, num_kv_heads, block_size, max_blocks_per_seq, num_seqs, query_half.data_ptr,
                      out_half.data_ptr, context_lens.data_ptr, k_cache.data_ptr, v_cache.data_ptr,
                      reinterpret_cast<int32_t*>(block_table.data_ptr), nullptr, is_causal, softmax_scale,
                      workspace_buffer.data_ptr, workspace_size, flashinfer_extra_workspace_buffer.data_ptr,
                      pinned_workspace_buffer.data_ptr, true, stream);

  helper_half.Forward();

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Validate half output
  ValidateAttentionOutput<half>(out_half, "FlashinferPrefillFP8E4M3Test_Half", stream);

  // Test with fp8_e4m3 KV cache type and bfloat16 scalar type
  FlashinferBatchPrefillHelper<__nv_bfloat16, llm_kernels::utils::KVCacheType::kFp8E4M3, __nv_bfloat16, int32_t>
      helper_bfloat16;

  helper_bfloat16.Prepare(num_heads, head_size, num_kv_heads, block_size, max_blocks_per_seq, num_seqs,
                          query_bfloat16.data_ptr, out_bfloat16.data_ptr, context_lens.data_ptr, k_cache.data_ptr,
                          v_cache.data_ptr, reinterpret_cast<int32_t*>(block_table.data_ptr), nullptr, is_causal,
                          softmax_scale, workspace_buffer.data_ptr, workspace_size,
                          flashinfer_extra_workspace_buffer.data_ptr, pinned_workspace_buffer.data_ptr, true, stream);

  helper_bfloat16.Forward();

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Validate bfloat16 output
  ValidateAttentionOutput<__nv_bfloat16>(out_bfloat16, "FlashinferPrefillFP8E4M3Test_BFloat16", stream);

  DeleteBuffer(query_half);
  DeleteBuffer(query_bfloat16);
  DeleteBuffer(out_half);
  DeleteBuffer(out_bfloat16);
  DeleteBuffer(k_cache);
  DeleteBuffer(v_cache);
  DeleteBuffer(block_table);
  DeleteBuffer(context_lens);
  DeleteBuffer(workspace_buffer);
  DeleteBuffer(flashinfer_extra_workspace_buffer);
  DeleteBuffer(pinned_workspace_buffer);
}
#endif
}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels