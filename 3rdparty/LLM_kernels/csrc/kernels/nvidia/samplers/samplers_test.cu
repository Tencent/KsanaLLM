/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <cstdlib>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/kernels/nvidia/samplers/repetition_penalty.h"
#include "csrc/kernels/nvidia/samplers/sampling_topk_kernels.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaSamplersTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T>
  uint32_t RefArgMax(const T* input_data, const size_t elem_num) {
    if (input_data == nullptr || elem_num == 0) {
      return 0;
    }
    T max_value = input_data[0];
    uint32_t max_index = 0;
    for (size_t idx = 0; idx < elem_num; ++idx) {
      if (max_value < input_data[idx]) {
        max_value = input_data[idx];
        max_index = idx;
      }
    }
    return max_index;
  }

  template <typename T>
  void TestGreedyCommon() {
    // create kernel's buffer
    int32_t batch_size = 2;
    int32_t vocab_size = 10;
    T max_logit = 5.0;
    T logit_range = 4.0;
    std::vector<uint32_t> base_result = {5, 7};

    // [batch_size, vocab_size]
    BufferMeta cpu_input =
        CreateBuffer<T>(MemoryType::MEMORY_CPU_PINNED,
                        {static_cast<size_t>(batch_size), static_cast<size_t>(vocab_size)}, true, 0, logit_range);
    T* intput_data = static_cast<T*>(cpu_input.data_ptr);
    for (size_t i = 0; i < base_result.size(); i++) {
      uint32_t index = i * vocab_size + base_result[i];
      intput_data[index] = max_logit;
    }
    BufferMeta input = CopyToDevice<T>(cpu_input);
    // [batch_size]
    BufferMeta result = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});

    InvokeArgMaxReduce<T>(static_cast<T*>(input.data_ptr), batch_size, vocab_size,
                          static_cast<uint32_t*>(result.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta cpu_result = CopyToHost<int32_t>(result);
    int32_t* cpu_result_ptr = static_cast<int32_t*>(cpu_result.data_ptr);
    for (int i = 0; i < batch_size; i++) {
      EXPECT_EQ(base_result[i], cpu_result_ptr[i]);
    }

    DeleteBuffer(cpu_result);
    DeleteBuffer(result);
    DeleteBuffer(input);
    DeleteBuffer(cpu_input);
  }

  template <typename T>
  void TestGreedyEqual() {
    if (std::getenv("ENABLE_NEW_ARGMAX") == nullptr) {
      return;
    }
    // create kernel's buffer
    int32_t batch_size = 3;
    int32_t vocab_size = 120;
    T max_logit = -0.5;
    // construct multiple maximum values for each batch
    std::vector<std::vector<uint32_t>> max_pos = {{1, 23}, {8, 87, 119}, {31, 45, 99, 100}};
    // When there are multiple maximum values, return the first one
    std::vector<uint32_t> base_result = {1, 8, 31};

    // [batch_size, vocab_size]
    BufferMeta cpu_input =
        CreateBuffer<T>(MemoryType::MEMORY_CPU_PINNED,
                        {static_cast<size_t>(batch_size), static_cast<size_t>(vocab_size)}, true, -5.0, -1.0);
    T* intput_data = static_cast<T*>(cpu_input.data_ptr);
    for (size_t i = 0; i < base_result.size(); i++) {
      for (size_t j = 0; j < max_pos[i].size(); j++) {
        uint32_t index = i * vocab_size + max_pos[i][j];
        intput_data[index] = max_logit;
      }
    }
    BufferMeta input = CopyToDevice<T>(cpu_input);
    // [batch_size]
    BufferMeta result = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});

    InvokeArgMaxReduce<T>(static_cast<T*>(input.data_ptr), batch_size, vocab_size,
                          static_cast<uint32_t*>(result.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta cpu_result = CopyToHost<int32_t>(result);
    int32_t* cpu_result_ptr = static_cast<int32_t*>(cpu_result.data_ptr);
    for (int i = 0; i < batch_size; i++) {
      EXPECT_EQ(base_result[i], cpu_result_ptr[i]);
    }

    DeleteBuffer(cpu_result);
    DeleteBuffer(result);
    DeleteBuffer(input);
    DeleteBuffer(cpu_input);
  }
};

TEST_F(LlamaNvidiaSamplersTestSuit, LlamaGreedyCommonTest) {
  TestGreedyCommon<float>();
  if (std::getenv("ENABLE_NEW_ARGMAX") == nullptr) {
    return;
  }
  TestGreedyCommon<half>();
  TestGreedyCommon<__nv_bfloat16>();
}

TEST_F(LlamaNvidiaSamplersTestSuit, LlamaGreedyEqualTest) {
  TestGreedyEqual<float>();
  TestGreedyEqual<half>();
  TestGreedyEqual<__nv_bfloat16>();
}

TEST_F(LlamaNvidiaSamplersTestSuit, LlamaGreedyLargeVocabSizeTest) {
  using DataType = float;
  // prepare input data
  BufferMeta input_data;
  input_data.LoadNpy<DataType>("/tmp/tests/kernels/data/sampler/greedy/input_float.npy", MemoryType::MEMORY_GPU);
  int32_t batch_size = input_data.shape[0];
  int32_t vocab_size = input_data.shape[1];
  // prepare output data
  BufferMeta result = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});

  InvokeArgMaxReduce<DataType>(static_cast<DataType*>(input_data.data_ptr), batch_size, vocab_size,
                               static_cast<uint32_t*>(result.data_ptr), stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  BufferMeta cpu_result = CopyToHost<int32_t>(result);
  int32_t* cpu_result_ptr = static_cast<int32_t*>(cpu_result.data_ptr);
  BufferMeta input_data_host = CopyToHost<DataType>(input_data);
  for (int32_t i = 0; i < batch_size; i++) {
    EXPECT_EQ(
        cpu_result_ptr[i],
        RefArgMax<DataType>(reinterpret_cast<const DataType*>(input_data_host.data_ptr) + i * vocab_size, vocab_size));
  }

  DeleteBuffer(input_data_host);
  DeleteBuffer(cpu_result);
  DeleteBuffer(result);
  DeleteBuffer(input_data);
}

TEST_F(LlamaNvidiaSamplersTestSuit, InvokeTopKSampling) {
  using DataType = float;
  void* workspace = nullptr;
  size_t workspaceSize = 0;
  BufferMeta logProbs;
  logProbs.LoadNpy<DataType>("/tmp/tests/kernels/data/sampler/greedy/input_float_10x32000.npy", MemoryType::MEMORY_GPU);
  int32_t batch_size = logProbs.shape[0];
  int32_t vocab_size = logProbs.shape[1];
  int32_t maxTopK = std::min(batch_size, vocab_size);  // Set the top K value
  BufferMeta topKs = CreateBuffer<int32_t>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  int32_t* topKs_ptr = static_cast<int32_t*>(topKs.data_ptr);
  for (int32_t i = 0; i < batch_size; i++) {
    topKs_ptr[i] = std::min(i + 1, maxTopK);
  }
  BufferMeta d_topKs = CopyToDevice<int32_t>(topKs);
  BufferMeta randomSeeds = CreateBuffer<uint64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});
  curandState_t* d_state;
  cudaMalloc(&d_state, batch_size * sizeof(curandState_t));
  tensorrt_llm::kernels::InvokeCurandBatchInitialize(d_state, nullptr, batch_size,
                                                     static_cast<uint64_t*>(randomSeeds.data_ptr), 0);
  float topP = 1;  // Set the top P value

  bool normalizeLogProbs = false;  // Set whether to normalize log probabilities
  bool logitsHasProbs = false;     // Set whether logits already have probabilities
  BufferMeta ids = CreateBuffer<uintptr_t>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  int** ids_ptr = reinterpret_cast<int**>(ids.data_ptr);
  BufferMeta d_id_vec = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});
  int32_t* d_id_vec_ptr = static_cast<int32_t*>(d_id_vec.data_ptr);
  for (int32_t i = 0; i < batch_size; i++) {
    ids_ptr[i] = reinterpret_cast<int*>(d_id_vec_ptr + i);
  }
  BufferMeta d_ids = CopyToDevice<uintptr_t>(ids);

  tensorrt_llm::kernels::invoke_batch_topk_sampling(
      workspace, workspaceSize, static_cast<float*>(logProbs.data_ptr), reinterpret_cast<int**>(d_ids.data_ptr),
      nullptr, nullptr, nullptr, nullptr, nullptr, d_state, maxTopK, static_cast<int*>(d_topKs.data_ptr), topP, nullptr,
      vocab_size, nullptr, nullptr, 0, batch_size, 0, nullptr, normalizeLogProbs, logitsHasProbs);
  cudaMalloc(&workspace, workspaceSize);
  tensorrt_llm::kernels::invoke_batch_topk_sampling(
      workspace, workspaceSize, static_cast<float*>(logProbs.data_ptr), reinterpret_cast<int**>(d_ids.data_ptr),
      nullptr, nullptr, nullptr, nullptr, nullptr, d_state, maxTopK, static_cast<int*>(d_topKs.data_ptr), topP, nullptr,
      vocab_size, nullptr, nullptr, 0, batch_size, 0, nullptr, normalizeLogProbs, logitsHasProbs);
  cudaFree(workspace);
  cudaFree(d_state);
  BufferMeta h_ids = CopyToHost<int32_t>(d_id_vec);
  int32_t* h_ids_ptr = static_cast<int32_t*>(h_ids.data_ptr);
  std::vector<int32_t> result = {29871, 338, 338, 29873, 29873, 413, 413, 29872, 29872, 29872};
  for (int32_t i = 0; i < batch_size; i++) {
    EXPECT_EQ(h_ids_ptr[i], result[i]);
  }

  DeleteBuffer(h_ids);
  DeleteBuffer(d_ids);
  DeleteBuffer(d_id_vec);
  DeleteBuffer(ids);
  DeleteBuffer(randomSeeds);
  DeleteBuffer(d_topKs);
  DeleteBuffer(topKs);
  DeleteBuffer(logProbs);
}

TEST_F(LlamaNvidiaSamplersTestSuit, InvokeTopKTopPSampling) {
  using DataType = float;
  void* workspace = nullptr;
  size_t workspaceSize = 0;
  BufferMeta logProbs;
  logProbs.LoadNpy<DataType>("/tmp/tests/kernels/data/sampler/greedy/input_float_10x32000.npy", MemoryType::MEMORY_GPU);
  int32_t batch_size = logProbs.shape[0];
  int32_t vocab_size = logProbs.shape[1];
  int32_t maxTopK = std::min(batch_size, vocab_size);  // Set the top K value
  BufferMeta topKs = CreateBuffer<int32_t>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  BufferMeta topPs = CreateBuffer<float>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  int32_t* topKs_ptr = static_cast<int32_t*>(topKs.data_ptr);
  float* topPs_ptr = static_cast<float*>(topPs.data_ptr);
  BufferMeta randomSeeds = CreateBuffer<uint64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});
  curandState_t* d_state;
  cudaMalloc(&d_state, batch_size * sizeof(curandState_t));
  tensorrt_llm::kernels::InvokeCurandBatchInitialize(d_state, nullptr, batch_size,
                                                     static_cast<uint64_t*>(randomSeeds.data_ptr), 0);

  bool normalizeLogProbs = false;  // Set whether to normalize log probabilities
  bool logitsHasProbs = true;      // Set whether logits already have probabilities
  BufferMeta ids = CreateBuffer<uintptr_t>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  BufferMeta batch_slots = CreateBuffer<int32_t>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  int** ids_ptr = reinterpret_cast<int**>(ids.data_ptr);
  BufferMeta d_id_vec = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size)});
  BufferMeta temperatures = CreateBuffer<float>(MemoryType::MEMORY_CPU_PINNED, {static_cast<size_t>(batch_size)});
  int32_t* d_id_vec_ptr = static_cast<int32_t*>(d_id_vec.data_ptr);
  for (int32_t i = 0; i < batch_size; i++) {
    ids_ptr[i] = reinterpret_cast<int*>(d_id_vec_ptr + i);
    int offset = (i + 2) % batch_size;
    static_cast<int32_t*>(batch_slots.data_ptr)[i] = offset;
    topKs_ptr[offset] = std::min(i + 1, maxTopK);
    topPs_ptr[offset] = 1.0 - (i / 10.0);
    static_cast<float*>(temperatures.data_ptr)[offset] = (i + 0.5) * 2.0;
  }
  BufferMeta d_ids = CopyToDevice<uintptr_t>(ids);
  BufferMeta d_batch_slots = CopyToDevice<int32_t>(batch_slots);
  BufferMeta d_temperatures = CopyToDevice<float>(temperatures);
  BufferMeta d_topKs = CopyToDevice<int32_t>(topKs);
  BufferMeta d_topPs = CopyToDevice<int32_t>(topPs);
  tensorrt_llm::kernels::InvokeAddBiasSoftMax<float>(static_cast<float*>(logProbs.data_ptr), nullptr,
                                                     static_cast<float*>(d_temperatures.data_ptr), nullptr, nullptr,
                                                     nullptr, static_cast<int32_t*>(d_batch_slots.data_ptr), batch_size,
                                                     0, 1, vocab_size, vocab_size, false, false, nullptr);

  tensorrt_llm::kernels::invoke_batch_topk_sampling(
      workspace, workspaceSize, static_cast<float*>(logProbs.data_ptr), reinterpret_cast<int**>(d_ids.data_ptr),
      nullptr, nullptr, nullptr, nullptr, nullptr, d_state, maxTopK, static_cast<int*>(d_topKs.data_ptr), 1.0,
      static_cast<float*>(d_topPs.data_ptr), vocab_size, nullptr, static_cast<int32_t*>(d_batch_slots.data_ptr), 0,
      batch_size, 0, nullptr, normalizeLogProbs, logitsHasProbs);
  cudaMalloc(&workspace, workspaceSize);
  tensorrt_llm::kernels::invoke_batch_topk_sampling(
      workspace, workspaceSize, static_cast<float*>(logProbs.data_ptr), reinterpret_cast<int**>(d_ids.data_ptr),
      nullptr, nullptr, nullptr, nullptr, nullptr, d_state, maxTopK, static_cast<int*>(d_topKs.data_ptr), 1.0,
      static_cast<float*>(d_topPs.data_ptr), vocab_size, nullptr, static_cast<int32_t*>(d_batch_slots.data_ptr), 0,
      batch_size, 0, nullptr, normalizeLogProbs, logitsHasProbs);
  BufferMeta h_ids = CopyToHost<int32_t>(d_id_vec);
  int32_t* h_ids_ptr = static_cast<int32_t*>(h_ids.data_ptr);
  std::vector<int32_t> result = {338, 29871, 29871, 338, 338, 338, 29873, 29873, 338, 338};
  for (int32_t i = 0; i < batch_size; i++) {
    EXPECT_EQ(h_ids_ptr[i], result[i]);
  }
  cudaFree(workspace);
  cudaFree(d_state);

  DeleteBuffer(h_ids);
  DeleteBuffer(d_topPs);
  DeleteBuffer(d_topKs);
  DeleteBuffer(d_temperatures);
  DeleteBuffer(d_batch_slots);
  DeleteBuffer(d_ids);
  DeleteBuffer(temperatures);
  DeleteBuffer(d_id_vec);
  DeleteBuffer(batch_slots);
  DeleteBuffer(ids);
  DeleteBuffer(randomSeeds);
  DeleteBuffer(topPs);
  DeleteBuffer(topKs);
  DeleteBuffer(logProbs);
}

TEST_F(LlamaNvidiaSamplersTestSuit, InvokeRepetitionPenaltyTest) {
  using DataType = float;
  const std::vector<size_t> test_data_size = {1ul, 1024ul, 32768ul};
  for (size_t vocab_size : test_data_size) {
    BufferMeta logits = CreateBuffer<DataType>(MemoryType::MEMORY_GPU, {vocab_size}, true);
    BufferMeta repetition_penalties = CreateBuffer<DataType>(MemoryType::MEMORY_GPU, {vocab_size}, true);
    BufferMeta output = CreateBuffer<DataType>(MemoryType::MEMORY_GPU, {vocab_size}, true);

    BufferMeta logits_host = CopyToHost<DataType>(logits);
    BufferMeta repetition_penalties_host = CopyToHost<DataType>(repetition_penalties);
    BufferMeta output_ref = CopyToHost<DataType>(output);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    DataType* logits_ptr = reinterpret_cast<DataType*>(logits_host.data_ptr);
    DataType* repetition_penalties_ptr = reinterpret_cast<DataType*>(repetition_penalties_host.data_ptr);
    DataType* output_ptr = reinterpret_cast<DataType*>(output_ref.data_ptr);
    for (size_t i = 0; i < vocab_size; ++i) {
      output_ptr[i] = logits_ptr[i] > 0 ? (logits_ptr[i] / repetition_penalties_ptr[i])
                                        : (logits_ptr[i] * repetition_penalties_ptr[i]);
    }
    BufferMeta output_ref_device = CopyToDevice<DataType>(output_ref);
    InvokeRepetitionPenalty(static_cast<const DataType*>(logits.data_ptr),
                            static_cast<const DataType*>(repetition_penalties.data_ptr),
                            static_cast<DataType*>(output.data_ptr), vocab_size, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_TRUE(CheckResult<DataType>("get_repetition_penalty_float_vocab_size_" + std::to_string(vocab_size),
                                      output_ref_device, output, 1e-5f, 1e-5f));
    DeleteBuffer(logits);
    DeleteBuffer(repetition_penalties);
    DeleteBuffer(output);
    DeleteBuffer(logits_host);
    DeleteBuffer(repetition_penalties_host);
    DeleteBuffer(output_ref);
    DeleteBuffer(output_ref_device);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
