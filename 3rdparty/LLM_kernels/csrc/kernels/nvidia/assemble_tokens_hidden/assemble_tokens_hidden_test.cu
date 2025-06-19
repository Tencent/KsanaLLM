/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"

#include <iostream>

namespace llm_kernels {
namespace nvidia {
namespace test {

template <typename T>
__global__ void InitInputRefOutputKernel(T* input_ptr, T* ref_output_ptr, const size_t total_ids_num,
                                         const size_t hidden_units) {
  constexpr float scale_factor = 1000000.0f;

  for (size_t glb_thread_idx = blockIdx.x * blockDim.x + threadIdx.x; glb_thread_idx < (total_ids_num * hidden_units);
       glb_thread_idx += blockDim.x) {
    input_ptr[glb_thread_idx] = (T)((glb_thread_idx + 1) / scale_factor);

    if (glb_thread_idx / hidden_units == 1 || glb_thread_idx / hidden_units == 4) {
      if (glb_thread_idx / hidden_units == 1) {
        ref_output_ptr[glb_thread_idx % hidden_units] = (T)((glb_thread_idx + 1) / scale_factor);
      }
      if (glb_thread_idx / hidden_units == 4) {
        ref_output_ptr[hidden_units + glb_thread_idx % hidden_units] = (T)((glb_thread_idx + 1) / scale_factor);
      }
    }
  }
}

class LlamaNvidiaAssembleTokensHiddenTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  const size_t warmup_times = 10;
  const size_t profile_run_times = 10;

  const size_t hidden_units = 4096ul;
  const int32_t max_length = 8;
  const int32_t batch_size = 2;
  const int32_t input_prompt_num = 2;
  const std::vector<std::vector<int32_t>> input_prompt_token_ids = {{1, 2}, {1024, 3, 0}};
  std::vector<size_t> ids_lens;

  template <typename T>
  void PrepareInputsOutputs(BufferMeta& input_meta, BufferMeta& ref_output_meta, size_t total_ids_num) {
    constexpr size_t block_size = 512ul;
    dim3 grid(total_ids_num);
    dim3 block(block_size);
    InitInputRefOutputKernel<T>
        <<<grid, block>>>((T*)input_meta.data_ptr, (T*)ref_output_meta.data_ptr, total_ids_num, hidden_units);
  }

  template <typename T>
  void CommonTest() {
    cudaEvent_t start_evt, stop_evt;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start_evt));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop_evt));

    // input batch size is 2
    // prompt 1 tensor shape [2, 4096]
    // [[0.1, 0.2, ..., 409.6],
    //  [409.7, 409.8, ..., 819.2]]
    // prompt 2 tensor shape [3, 4096]
    // [[819.3, 819.4, ..., 1228.8],
    //  [1228.9, 1229.0, ..., 1638.4],
    //  [1638.5, 1638.6, ..., 2048]]
    // output
    // [[409.7, 409.8, ..., 819.2],
    //  [1638.5, 1638.6, ..., 2048]]

    BufferMeta input_meta = CreateBuffer<T>(
        MemoryType::MEMORY_GPU, {input_prompt_token_ids[0].size() + input_prompt_token_ids[1].size(), hidden_units});

    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size), hidden_units});
    BufferMeta ref_output_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size), hidden_units});

    BufferMeta ids_index =
        CreateBuffer<size_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(batch_size), hidden_units});

    ids_lens.assign(batch_size, 0ul);
    size_t total_ids_num = 0ul;
    for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
      ids_lens[prompt_id] = total_ids_num + input_prompt_token_ids[prompt_id].size() - 1;
      total_ids_num += input_prompt_token_ids[prompt_id].size();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(reinterpret_cast<size_t*>(ids_index.data_ptr), ids_lens.data(),
                                       sizeof(size_t) * ids_lens.size(), cudaMemcpyHostToDevice));
    PrepareInputsOutputs<T>(input_meta, ref_output_meta, total_ids_num);
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t i = 0; i < warmup_times; ++i) {
      // Execute AssembleTokensHidden kernel warmup
      AssembleTokensHidden(reinterpret_cast<const T*>(input_meta.data_ptr),
                        reinterpret_cast<const size_t*>(ids_index.data_ptr), batch_size, hidden_units,
                        reinterpret_cast<T*>(output_meta.data_ptr), stream);
    }

    // Record start event
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start_evt, stream));
    for (size_t i = 0; i < profile_run_times; ++i) {
      AssembleTokensHidden(reinterpret_cast<const T*>(input_meta.data_ptr),
                        reinterpret_cast<const size_t*>(ids_index.data_ptr), batch_size, hidden_units,
                        reinterpret_cast<T*>(output_meta.data_ptr), stream);
    }
    // Record end event
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop_evt, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Calculate and print execution time
    float elapsed_time_ms = 0.0f;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time_ms, start_evt, stop_evt));
    elapsed_time_ms /= profile_run_times;
    std::cout << "AssembleTokensHidden execution time: " << elapsed_time_ms << " ms" << std::endl;

    // Calculate theoretical performance of AssembleTokensHidden
    // Each element requires 2 memory operations (1 read + 1 write)
    size_t bytes_processed =
        sizeof(T) * (input_prompt_token_ids[0].size() + input_prompt_token_ids[1].size() + batch_size) * hidden_units;
    float kernel_bandwidth_gbps = (bytes_processed / (1024.0f * 1024.0f * 1024.0f)) / (elapsed_time_ms / 1000.0f);
    std::cout << "AssembleTokensHidden bandwidth: " << kernel_bandwidth_gbps << " GB/s" << std::endl;

    void* output_half_host;
    void* ref_output_host;
    CHECK_NVIDIA_CUDA_ERROR(cudaMallocHost(&output_half_host, sizeof(T) * batch_size * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(cudaMallocHost(&ref_output_host, sizeof(T) * batch_size * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(reinterpret_cast<T*>(output_half_host),
                                       reinterpret_cast<T*>(output_meta.data_ptr),
                                       sizeof(T) * batch_size * hidden_units, cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(reinterpret_cast<T*>(ref_output_host),
                                       reinterpret_cast<T*>(ref_output_meta.data_ptr),
                                       sizeof(T) * batch_size * hidden_units, cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    for (size_t idx = 0; idx < batch_size * hidden_units; ++idx) {
      float output_f_value = (float)(reinterpret_cast<T*>(output_half_host)[idx]);
      float ref_output_f_value = (float)(reinterpret_cast<T*>(ref_output_host)[idx]);
      EXPECT_TRUE((output_f_value - ref_output_f_value) < 1e-4)
          << "Fail in idx: " << idx << " output value: " << output_f_value << ", ref_value: " << ref_output_f_value;
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(ref_output_host));
    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(output_half_host));

    // Destroy CUDA events
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start_evt));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop_evt));
  }
};

TEST_F(LlamaNvidiaAssembleTokensHiddenTestSuit, FloatCommonTest) { CommonTest<float>(); }

TEST_F(LlamaNvidiaAssembleTokensHiddenTestSuit, Float16CommonTest) { CommonTest<half>(); }

TEST_F(LlamaNvidiaAssembleTokensHiddenTestSuit, BFloat16CommonTest) { CommonTest<__nv_bfloat16>(); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
