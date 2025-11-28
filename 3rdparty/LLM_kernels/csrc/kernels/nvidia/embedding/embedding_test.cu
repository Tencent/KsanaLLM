/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

template <typename T>
__global__ void InitTablesKernel(T* emb_table_ptr, T* pos_table_ptr) {
  uint32_t glb_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  emb_table_ptr[glb_thread_idx] = (T)(glb_thread_idx / 10000.0f);
  pos_table_ptr[glb_thread_idx] = (T)(glb_thread_idx / 10000.0f);
}

class LlamaNvidiaEmbeddingTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  // mock qwen3-32b vocab size
  const size_t vocab_size = 152064ul;
  const size_t vocab_id = 0ul;
  const size_t hidden_units = 8192ul;
  const int32_t max_length = 8;
  const int32_t batch_size = 2;
  const int32_t input_prompt_num = 2;
  // Test input prompts containing token IDs for embedding lookup testing
  // First prompt: sequential token IDs from 1-110, second prompt: sparse token IDs {1042,4,12,41}
  const std::vector<std::vector<int32_t>> input_prompt_token_ids = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110}, {1042,4,12,41}};
  std::vector<size_t> ids_lens;

  template <typename T>
  void PrepareTables(BufferMeta& emb_table_meta, BufferMeta& pos_table_meta) {
    size_t total_nums = vocab_size * hidden_units;
    size_t block_size = 512ul;
    dim3 grid((total_nums + block_size - 1) / block_size);
    dim3 block(block_size);
    InitTablesKernel<T><<<grid, block, 0, stream>>>((T*)emb_table_meta.data_ptr, (T*)pos_table_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
};

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRHalfTest) {
  // emb table shape: [4, 4096]
  // [[0.1, 0.2, ..., 409.6],
  //  [409.7, 409.8, ..., 819.2],
  //  ...
  //  [1228.9, 1229.0, ..., 1638.4]]
  BufferMeta emb_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  // pos table shape: [4, 4096]
  // [[0.001, 0.002, ..., 4.096],
  //  [4.097, 4.098, ..., 8.192],
  //  ...
  //  [12.289, 12.290, ..., 16.384]]
  BufferMeta pos_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  PrepareTables<half>(emb_table_meta, pos_table_meta);

  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }
  std::vector<size_t> steps_host(total_ids_num, 0ul);
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    std::iota(steps_host.data() + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx + 1],
              0ul);
  }

  int32_t* input_ids;
  size_t* steps;
  size_t* ids_offsets;
  size_t* prefix_offsets;
  half* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(float) * total_ids_num * hidden_units));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                   sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(steps + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx],
                   sizeof(size_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(),
                                     cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * ids_offsets_host.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, total_ids_num * hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  const float emb_scale = 1.f;
  LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ true, /* USE_EMB_SCALE */ false>(
      reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
      reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
      InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, batch_size,
      hidden_units, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    half* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMallocHost(&host_result_ptr, sizeof(half) * input_prompt_token_ids[prompt_id].size() * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units,
                   sizeof(half) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units,
                   cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // compute refer
    std::vector<float> host_ref_result_vec(input_prompt_token_ids[prompt_id].size() * hidden_units, 0.0f);
    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];
      size_t step = steps_host[ids_offsets_host[prompt_id] + prompt_token_id];

      for (size_t emb_id = 0; emb_id < hidden_units; ++emb_id) {
        half ref_emb_value = static_cast<half>((token_id * hidden_units + emb_id) / 10000.0f);
        half ref_pos_emb_value = static_cast<half>((step * hidden_units + emb_id) / 10000.0f);
        // When use_emb_scale is true and DO_POSITION_ENCODING is true, the formula is: emb_vec_val * emb_scale + pos_emb_vec_val
        half ref_value = ref_emb_value * static_cast<half>(emb_scale) + ref_pos_emb_value;
        if (static_cast<size_t>(token_id) >= vocab_size) {
          ref_value = static_cast<half>(0.0f);
        }
        half result_value = host_result_ptr[prompt_token_id * hidden_units + emb_id];
        EXPECT_TRUE((ref_value - result_value) < static_cast<half>(1e-3))
            << "Fail in token: " << token_id << " emb: " << emb_id << ", result_value: " << (float)result_value
            << ", ref_value: " << (float)ref_value << ", emb_scale: " << emb_scale;
      }
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRHalfMisAlignedTest) {
  // Use hidden_units that is not divisible by 4 or 8 to exercise scalar tail
  const size_t hidden_units_tail = 4097ul;

  BufferMeta emb_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units_tail});
  BufferMeta pos_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units_tail});

  // Initialize tables
  {
    size_t total_nums = vocab_size * hidden_units_tail;
    size_t block_size = 512ul;
    dim3 grid((total_nums + block_size - 1) / block_size);
    dim3 block(block_size);
    InitTablesKernel<half><<<grid, block, 0, stream>>>((half*)emb_table_meta.data_ptr, (half*)pos_table_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }
  std::vector<size_t> steps_host(total_ids_num, 0ul);
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    std::iota(steps_host.data() + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx + 1], 0ul);
  }

  int32_t* input_ids;
  size_t* steps;
  size_t* ids_offsets;
  size_t* prefix_offsets;
  half* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(half) * total_ids_num * hidden_units_tail));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                                       sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(steps + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx],
                                       sizeof(size_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * ids_offsets_host.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, sizeof(half) * total_ids_num * hidden_units_tail));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  const float emb_scale = 1.f;
  LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ true, /* USE_EMB_SCALE */ false>(
      reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
      reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
      InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, batch_size,
      hidden_units_tail, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    half* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMallocHost(&host_result_ptr, sizeof(half) * input_prompt_token_ids[prompt_id].size() * hidden_units_tail));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units_tail,
                                       sizeof(half) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units_tail,
                                       cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];
      size_t step = steps_host[ids_offsets_host[prompt_id] + prompt_token_id];

      for (size_t emb_id = 0; emb_id < hidden_units_tail; ++emb_id) {
        half ref_emb_value = static_cast<half>((token_id * hidden_units_tail + emb_id) / 10000.0f);
        half ref_pos_emb_value = static_cast<half>((step * hidden_units_tail + emb_id) / 10000.0f);
        half ref_value = ref_emb_value * static_cast<half>(emb_scale) + ref_pos_emb_value;
        if (static_cast<size_t>(token_id) >= vocab_size) {
          ref_value = static_cast<half>(0.0f);
        }
        half result_value = host_result_ptr[prompt_token_id * hidden_units_tail + emb_id];
        EXPECT_TRUE((ref_value - result_value) < static_cast<half>(1e-3))
            << "Fail in token: " << token_id << " emb: " << emb_id << ", result_value: " << (float)result_value
            << ", ref_value: " << (float)ref_value << ", emb_scale: " << emb_scale;
      }
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRHalfTestWithEmbScale) {
  // emb table shape: [4, 4096]
  // [[0.1, 0.2, ..., 409.6],
  //  [409.7, 409.8, ..., 819.2],
  //  ...
  //  [1228.9, 1229.0, ..., 1638.4]]
  BufferMeta emb_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  // pos table shape: [4, 4096]
  // [[0.001, 0.002, ..., 4.096],
  //  [4.097, 4.098, ..., 8.192],
  //  ...
  //  [12.289, 12.290, ..., 16.384]]
  BufferMeta pos_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  const float emb_scale = 12.f;
  PrepareTables<half>(emb_table_meta, pos_table_meta);

  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }
  std::vector<size_t> steps_host(total_ids_num, 0ul);
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    std::iota(steps_host.data() + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx + 1],
              0ul);
  }

  int32_t* input_ids;
  size_t* steps;
  size_t* ids_offsets;
  size_t* prefix_offsets;
  half* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(float) * total_ids_num * hidden_units));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                   sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(steps + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx],
                   sizeof(size_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(),
                                     cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * ids_offsets_host.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, total_ids_num * hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ false, /* USE_EMB_SCALE */ true>(
      reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
      reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
      InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, batch_size,
      hidden_units, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    half* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMallocHost(&host_result_ptr, sizeof(half) * input_prompt_token_ids[prompt_id].size() * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units,
                   sizeof(half) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units,
                   cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // compute refer
    std::vector<float> host_ref_result_vec(input_prompt_token_ids[prompt_id].size() * hidden_units, 0.0f);
    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];

      for (size_t emb_id = 0; emb_id < hidden_units; ++emb_id) {
        half ref_emb_value = static_cast<half>((token_id * hidden_units + emb_id) / 10000.0f);
        // When use_emb_scale is true and DO_POSITION_ENCODING is false, the formula is: emb_vec_val * emb_scale
        half ref_value = ref_emb_value * static_cast<half>(emb_scale);
        if (static_cast<size_t>(token_id) >= vocab_size) {
          ref_value = static_cast<half>(0.0f);
        }
        half result_value = host_result_ptr[prompt_token_id * hidden_units + emb_id];
        EXPECT_TRUE((ref_value - result_value) < static_cast<half>(1e-3))
            << "Fail in token: " << token_id << " emb: " << emb_id << ", result_value: " << (float)result_value
            << ", ref_value: " << (float)ref_value << ", emb_scale: " << emb_scale;
      }
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRFloatAccuracyTest) {
  BufferMeta emb_table_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  BufferMeta pos_table_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  PrepareTables<float>(emb_table_meta, pos_table_meta);

  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }
  std::vector<size_t> steps_host(total_ids_num, 0ul);
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    std::iota(steps_host.data() + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx + 1], 0ul);
  }

  int32_t* input_ids;
  size_t* steps;
  size_t* ids_offsets;
  size_t* prefix_offsets;
  float* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(float) * total_ids_num * hidden_units));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                                       sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(steps + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx],
                                       sizeof(size_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * ids_offsets_host.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, total_ids_num * hidden_units * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  const float emb_scale = 1.f;
  LookupFusedEmbeddingWithCSRInputs<float, /* DO_POSITION_ENCODING */ true, /* USE_EMB_SCALE */ false>(
      reinterpret_cast<float*>(output_hidden_units), reinterpret_cast<const float*>(emb_table_meta.data_ptr),
      reinterpret_cast<const float*>(pos_table_meta.data_ptr), static_cast<float>(emb_scale),
      InvokeInputIdsEmbeddingLookupPosEncodingParam<float>{}, input_ids, steps, ids_offsets, prefix_offsets, batch_size,
      hidden_units, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    float* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMallocHost(&host_result_ptr, sizeof(float) * input_prompt_token_ids[prompt_id].size() * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units,
                                       sizeof(float) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units,
                                       cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];
      size_t step = steps_host[ids_offsets_host[prompt_id] + prompt_token_id];
      for (size_t emb_id = 0; emb_id < hidden_units; ++emb_id) {
        float ref_emb_value = static_cast<float>((token_id * hidden_units + emb_id) / 10000.0f);
        float ref_pos_emb_value = static_cast<float>((step * hidden_units + emb_id) / 10000.0f);
        float ref_value = ref_emb_value * emb_scale + ref_pos_emb_value;
        if (static_cast<size_t>(token_id) >= vocab_size) {
          ref_value = 0.0f;
        }
        float result_value = host_result_ptr[prompt_token_id * hidden_units + emb_id];
        EXPECT_NEAR(ref_value, result_value, 1e-3) << "Fail in token: " << token_id << " emb: " << emb_id
                                                  << ", result_value: " << result_value << ", ref_value: " << ref_value
                                                  << ", emb_scale: " << emb_scale;
      }
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRFloatAccuracyTestWithEmbScale) {
  BufferMeta emb_table_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  BufferMeta pos_table_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  const float emb_scale = 12.f;
  PrepareTables<float>(emb_table_meta, pos_table_meta);

  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }
  std::vector<size_t> steps_host(total_ids_num, 0ul);
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    std::iota(steps_host.data() + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx + 1], 0ul);
  }

  int32_t* input_ids;
  size_t* steps;
  size_t* ids_offsets;
  size_t* prefix_offsets;
  float* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(float) * total_ids_num * hidden_units));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                                       sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(steps + ids_offsets_host[tokens_idx], steps_host.data() + ids_offsets_host[tokens_idx],
                                       sizeof(size_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * ids_offsets_host.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, total_ids_num * hidden_units * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  LookupFusedEmbeddingWithCSRInputs<float, /* DO_POSITION_ENCODING */ false, /* USE_EMB_SCALE */ true>(
      reinterpret_cast<float*>(output_hidden_units), reinterpret_cast<const float*>(emb_table_meta.data_ptr),
      reinterpret_cast<const float*>(pos_table_meta.data_ptr), static_cast<float>(emb_scale),
      InvokeInputIdsEmbeddingLookupPosEncodingParam<float>{}, input_ids, steps, ids_offsets, prefix_offsets, batch_size,
      hidden_units, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    float* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMallocHost(&host_result_ptr, sizeof(float) * input_prompt_token_ids[prompt_id].size() * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units,
                                       sizeof(float) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units,
                                       cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];
      for (size_t emb_id = 0; emb_id < hidden_units; ++emb_id) {
        float ref_emb_value = static_cast<float>((token_id * hidden_units + emb_id) / 10000.0f);
        float ref_value = ref_emb_value * emb_scale;
        if (static_cast<size_t>(token_id) >= vocab_size) {
          ref_value = 0.0f;
        }
        float result_value = host_result_ptr[prompt_token_id * hidden_units + emb_id];
        EXPECT_NEAR(ref_value, result_value, 1e-3) << "Fail in token: " << token_id << " emb: " << emb_id
                                                  << ", result_value: " << result_value << ", ref_value: " << ref_value
                                                  << ", emb_scale: " << emb_scale;
      }
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

TEST_F(LlamaNvidiaEmbeddingTestSuit, DISABLED_LlamaFusedEmbeddingPerfTest) {
  // Allocate and initialize tables once
  BufferMeta emb_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  BufferMeta pos_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  PrepareTables<half>(emb_table_meta, pos_table_meta);

  const float emb_scale = 1.f;
  const std::vector<int> token_nums = {1, 16, 32, 64, 128, 256, 512};
  const std::vector<int> batch_sizes = {1, 4, 8, 16, 32, 64, 128};

  for (int bs : batch_sizes) {
    for (int token_num : token_nums) {
    // Build CSR offsets for equal-length prompts
    std::vector<size_t> ids_offsets_host(bs + 1, 0ul);
    for (int b = 0; b < bs; ++b) {
      ids_offsets_host[b + 1] = ids_offsets_host[b] + static_cast<size_t>(token_num);
    }
    const size_t total_ids_num = ids_offsets_host.back();

    // Host inputs
    std::vector<int32_t> input_ids_host(total_ids_num, 0);
    std::vector<size_t> steps_host(total_ids_num, 0ul);
    for (int b = 0; b < bs; ++b) {
      for (int t = 0; t < token_num; ++t) {
        size_t idx = ids_offsets_host[b] + static_cast<size_t>(t);
        input_ids_host[idx] = 1 + (t % static_cast<int>(vocab_size - 1));
        steps_host[idx] = static_cast<size_t>(t);
      }
    }

    // Device buffers
    int32_t* input_ids = nullptr;
    size_t* steps = nullptr;
    size_t* ids_offsets = nullptr;
    size_t* prefix_offsets = nullptr;
    half* output_hidden_units = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&steps, sizeof(size_t) * total_ids_num));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (bs + 1)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&prefix_offsets, sizeof(size_t) * (bs + 1)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(half) * total_ids_num * hidden_units));

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(input_ids, input_ids_host.data(), sizeof(int32_t) * total_ids_num, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(steps, steps_host.data(), sizeof(size_t) * total_ids_num, cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * (bs + 1), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(prefix_offsets, 0x0, sizeof(size_t) * (bs + 1)));

    // Case 1: DO_POSITION_ENCODING=true, USE_EMB_SCALE=false
    {
      const int warmup_iters = 10;
      const int measure_iters = 200;
      for (int i = 0; i < warmup_iters; ++i) {
        LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ true, /* USE_EMB_SCALE */ false>(
            reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
            reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
            InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, bs,
            hidden_units, vocab_size, vocab_id, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      cudaEvent_t start_e, stop_e;
      CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start_e, stream));
      for (int i = 0; i < measure_iters; ++i) {
        LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ true, /* USE_EMB_SCALE */ false>(
            reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
            reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
            InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, bs,
            hidden_units, vocab_size, vocab_id, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop_e, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop_e));
      float total_ms = 0.0f;
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&total_ms, start_e, stop_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop_e));
      double avg_ms = total_ms / static_cast<double>(measure_iters);
    }

    // Case 2: DO_POSITION_ENCODING=false, USE_EMB_SCALE=true
    {
      const int warmup_iters = 10;
      const int measure_iters = 200;
      for (int i = 0; i < warmup_iters; ++i) {
        LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ false, /* USE_EMB_SCALE */ true>(
            reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
            reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
            InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, bs,
            hidden_units, vocab_size, vocab_id, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      cudaEvent_t start_e, stop_e;
      CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start_e, stream));
      for (int i = 0; i < measure_iters; ++i) {
        LookupFusedEmbeddingWithCSRInputs<half, /* DO_POSITION_ENCODING */ false, /* USE_EMB_SCALE */ true>(
            reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
            reinterpret_cast<const half*>(pos_table_meta.data_ptr), static_cast<half>(emb_scale),
            InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{}, input_ids, steps, ids_offsets, prefix_offsets, bs,
            hidden_units, vocab_size, vocab_id, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop_e, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop_e));
      float total_ms = 0.0f;
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&total_ms, start_e, stop_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start_e));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop_e));
      double avg_ms = total_ms / static_cast<double>(measure_iters);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(prefix_offsets));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(steps));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
    }
  }

  DeleteBuffer(emb_table_meta);
  DeleteBuffer(pos_table_meta);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
