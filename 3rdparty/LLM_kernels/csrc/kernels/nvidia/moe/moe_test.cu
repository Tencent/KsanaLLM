/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "moe.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#define WARP_SIZE 32

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaMoeTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

  template <typename T>
  void SiluMulKernelAccTest();
  template <typename T>
  void SiluMulKernelPerformanceTest();

 protected:
  size_t inter_size = 1024;
  size_t topk = 8;
  using NvidiaTestSuitBase::stream;
  const std::vector<size_t> m_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
};

TEST_F(LlamaNvidiaMoeTestSuit, SumOutDim1KernelTest) {
  const int num_tokens = 20;
  const int num_experts = 256;
  const int topk = 8;
  const int hidden_size = 16;

  size_t total_elements = static_cast<size_t>(num_tokens) * topk * hidden_size;
  size_t output_elements = static_cast<size_t>(num_tokens) * hidden_size;
  std::vector<float> input(total_elements);
  std::vector<float> expected_output(output_elements);
  std::vector<float> output(output_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    input[i] = i;
    size_t num_token_i = i / topk / hidden_size;
    size_t hidden_size_i = i % hidden_size;
    expected_output[num_token_i * hidden_size + hidden_size_i] += i;
  }

  void* d_input;
  void* d_output;
  cudaMalloc(&d_input, total_elements * sizeof(float));
  cudaMalloc(&d_output, output_elements * sizeof(float));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_input, reinterpret_cast<void*>(input.data()), total_elements * sizeof(float),
                                     cudaMemcpyHostToDevice));

  for (int i = 0; i < 10; ++i)
    InvokeMoeSum<float, false>(d_input, d_output, nullptr, num_tokens, topk, hidden_size, stream);

  cudaEvent_t start, stop;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(output.data(), d_output, output_elements * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < output_elements; ++i) {
    EXPECT_NEAR(output[i], expected_output[i], 1e-5);
  }

  cudaFree(d_input);
  cudaFree(d_output);
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop));
}

template <typename T>
void LlamaNvidiaMoeTestSuit::SiluMulKernelAccTest() {
  std::string type_str = "float";
  float tol = 1e-5f;
  if (std::is_same<T, half>::value) {
    type_str = "half";
    tol = 1e-3f;  // half precision has higher tolerance
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    type_str = "bfloat16";
    tol = 1e-3f;  // __nv_bfloat16 precision has higher tolerance
  }
  for (size_t i = 0; i < m_list.size(); i++) {
    size_t m = m_list[i];
    size_t num_elements = m * topk * inter_size * 2;
    BufferMeta d_input = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m * topk, inter_size * 2},
                                         /*is_random_init*/ true);
    BufferMeta d_output_ref = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m * topk, inter_size},
                                              /*is_random_init*/ true);
    BufferMeta d_output_flashinfer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m * topk, inter_size},
                                                     /*is_random_init*/ true);

    SiluAndMul<T, false>(reinterpret_cast<const T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output_ref.data_ptr),
                         nullptr, num_elements, inter_size, stream);
    FlashinferSiluAndMul<T>(reinterpret_cast<const T*>(d_input.data_ptr),
                            reinterpret_cast<T*>(d_output_flashinfer.data_ptr), nullptr, num_elements, inter_size,
                            stream);

    EXPECT_TRUE(CheckResult<T>("SiluKernelTest dtype: " + type_str + " m = " + std::to_string(m), d_output_ref,
                               d_output_flashinfer, tol, tol));
    DeleteBuffer(d_input);
    DeleteBuffer(d_output_ref);
    DeleteBuffer(d_output_flashinfer);
  }
}

template <typename T>
void LlamaNvidiaMoeTestSuit::SiluMulKernelPerformanceTest() {
  for (size_t i = 0; i < m_list.size(); i++) {
    size_t m = m_list[i];
    size_t num_elements = m * topk * inter_size * 2;
    std::cout << "===== Testing with m = " << m << " =====" << std::endl;
    BufferMeta d_input = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m * topk, inter_size * 2},
                                         /*is_random_init*/ true);
    BufferMeta d_output = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m * topk, inter_size},
                                          /*is_random_init*/ true);

    auto cuda_run = [&]() {
      SiluAndMul<T, false>(reinterpret_cast<const T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output.data_ptr),
                           nullptr, num_elements, inter_size, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << std::setw(25) << "SiluAndMul "
              << "Kernel execution 1 times " << std::setw(10) << milliseconds << " ms" << std::endl;

    auto cuda_run_flashinfer = [&]() {
      FlashinferSiluAndMul<T>(reinterpret_cast<const T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output.data_ptr),
                              nullptr, num_elements, inter_size, stream);
    };
    milliseconds = MeasureCudaExecutionTime(cuda_run_flashinfer, stream, 10, 100);
    std::cout << std::left << std::setw(25) << "FlashinferSiluAndMul "
              << "Kernel execution 1 times " << std::setw(10) << milliseconds << " ms" << std::endl;

    DeleteBuffer(d_input);
    DeleteBuffer(d_output);
  }
}

TEST_F(LlamaNvidiaMoeTestSuit, halfSiluKernelTest) {
  SiluMulKernelAccTest<half>();
  SiluMulKernelPerformanceTest<half>();
}

TEST_F(LlamaNvidiaMoeTestSuit, FloatSiluKernelTest) {
  SiluMulKernelAccTest<float>();
  SiluMulKernelPerformanceTest<float>();
}

TEST_F(LlamaNvidiaMoeTestSuit, bf16SiluKernelTest) {
  SiluMulKernelAccTest<__nv_bfloat16>();
  SiluMulKernelPerformanceTest<__nv_bfloat16>();
}

void MoeAlignBlockCpu(const int* topk_ids, int* expert_ids, int* sorted_ids, int* token_post_pad, int token_num,
                      int topk, int expert_num, int block_size) {
  std::vector<int> cumsum(expert_num + 1);
  std::vector<int> token_cnts(expert_num);
  size_t numel = static_cast<size_t>(token_num) * topk;
  for (size_t i = 0; i < numel; ++i) {
    int expert_id = topk_ids[i];
    if (expert_id < 0) {
      continue;
    }
    token_cnts[expert_id] += 1;
  }
  for (int i = 0; i < expert_num; ++i) {
    cumsum[i + 1] = cumsum[i] + (token_cnts[i] + block_size - 1) / block_size;
    token_cnts[i] = 0;
    for (int j = cumsum[i]; j < cumsum[i + 1]; ++j) {
      expert_ids[j] = i;
    }
  }
  token_post_pad[0] = cumsum[expert_num] * block_size;
  for (size_t i = 0; i < numel; ++i) {
    int expert_id = topk_ids[i];
    if (expert_id < 0) {
      continue;
    }
    int idx = cumsum[expert_id] * block_size + token_cnts[expert_id];
    sorted_ids[idx] = i;
    token_cnts[expert_id] += 1;
  }
}

TEST_F(LlamaNvidiaMoeTestSuit, MoeAlignBlockKernelAccTest) {
  int token_num = 4;
  int topk = 6;
  int block_size = 64;
  int num_thread = 256;

  std::vector<int> expert_para_sizes = {1, 2, 4, 8};
  std::vector<int> expert_sizes = {8, 32, 64, 128, 256};

  for (int expert_para_size : expert_para_sizes) {
    for (int num_experts : expert_sizes) {
      int num_experts_per_rank = num_experts / expert_para_size;
      size_t numel = token_num * topk;
      int max_num_tokens_padded = numel + num_experts_per_rank * (block_size - 1);
      int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;

      // Check shared mem
      int shared_mem = std::max(
          /* vllm */ ((num_thread + 2) * num_experts) * sizeof(uint16_t) + (num_experts + 1) * sizeof(int32_t),
          /* sglang */ (num_experts + (num_experts + 1) + NextPow2(num_experts) + WARP_SIZE) * sizeof(int32_t));
      int device_max_shared_mem = getMaxSharedMemoryPerBlockOptin();
      if (device_max_shared_mem < shared_mem) {
        std::cout << "Current GPU Device do not support Shared Memory " << shared_mem
                  << ", cudaDevAttrMaxSharedMemoryPerBlockOptin = " << device_max_shared_mem << std::endl;
        continue;
      }

      // Prepare host data
      BufferMeta h_topk_ids =
          CreateBuffer<int>(MemoryType::MEMORY_CPU, {static_cast<size_t>(token_num), static_cast<size_t>(topk)},
                            /*is_random_init*/ true, /*min_val*/ -1, /*max_val*/ num_experts_per_rank - 1);
      BufferMeta h_sorted_token_ids =
          CreateBuffer<int>(MemoryType::MEMORY_CPU, {static_cast<size_t>(max_num_tokens_padded)},
                            /*is_random_init*/ true, /*min_val*/ numel, /*max_val*/ numel);
      BufferMeta h_expert_ids = CreateBuffer<int>(MemoryType::MEMORY_CPU, {static_cast<size_t>(max_num_m_blocks)},
                                                  /*is_random_init*/ false);
      BufferMeta h_total_tokens_post_pad = CreateBuffer<int>(MemoryType::MEMORY_CPU, {1},
                                                             /*is_random_init*/ false);

      // Invoke cpu version
      MoeAlignBlockCpu(reinterpret_cast<const int*>(h_topk_ids.data_ptr), reinterpret_cast<int*>(h_expert_ids.data_ptr),
                       reinterpret_cast<int*>(h_sorted_token_ids.data_ptr),
                       reinterpret_cast<int*>(h_total_tokens_post_pad.data_ptr), token_num, topk, num_experts_per_rank,
                       block_size);

      // Prepare device data
      BufferMeta d_topk_ids = CopyToDevice<int>(h_topk_ids);
      BufferMeta d_sorted_token_ids = CopyToDevice<int>(h_sorted_token_ids);
      BufferMeta d_expert_ids = CopyToDevice<int>(h_expert_ids);
      BufferMeta d_total_tokens_post_pad = CopyToDevice<int>(h_total_tokens_post_pad);

      // Invoke vllm version
      if (expert_para_size == 1) {
        InvokeMoeAlignBlockSize<int, uint16_t, false>(
            reinterpret_cast<int*>(d_topk_ids.data_ptr), reinterpret_cast<int*>(d_sorted_token_ids.data_ptr),
            reinterpret_cast<int*>(d_expert_ids.data_ptr), reinterpret_cast<int*>(d_total_tokens_post_pad.data_ptr),
            topk, num_experts_per_rank, expert_para_size, block_size, numel, 0, stream);
      } else {
        InvokeMoeAlignBlockSize<int, uint16_t, true>(
            reinterpret_cast<int*>(d_topk_ids.data_ptr), reinterpret_cast<int*>(d_sorted_token_ids.data_ptr),
            reinterpret_cast<int*>(d_expert_ids.data_ptr), reinterpret_cast<int*>(d_total_tokens_post_pad.data_ptr),
            topk, num_experts_per_rank, expert_para_size, block_size, numel, 0, stream);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      // Verify accuracy of vllm version
      EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                       std::to_string(num_experts) + "_expert_ids",
                                   h_expert_ids, d_expert_ids, 0, 0));
      EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                       std::to_string(num_experts) + "_sorted_token_ids",
                                   h_sorted_token_ids, d_sorted_token_ids, 0, 0));
      EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                       std::to_string(num_experts) + "_total_tokens_post_pad",
                                   h_total_tokens_post_pad, d_total_tokens_post_pad, 0, 0));

      if (expert_para_size == 1 && num_experts >= 128) {
        BufferMeta d_cumsum = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_experts + 1)},
                                                /*is_random_init*/ false);

        // Invoke sglang version
        InvokeSglMoeAlignBlockSize<int>(
            reinterpret_cast<int*>(d_topk_ids.data_ptr), reinterpret_cast<int*>(d_sorted_token_ids.data_ptr),
            reinterpret_cast<int*>(d_expert_ids.data_ptr), reinterpret_cast<int*>(d_total_tokens_post_pad.data_ptr),
            max_num_tokens_padded, num_experts, block_size, numel, reinterpret_cast<int*>(d_cumsum.data_ptr), stream);
        CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Verify accuracy of sglang version
        EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                         std::to_string(num_experts) + "_expert_ids",
                                     h_expert_ids, d_expert_ids, 0, 0));
        EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                         std::to_string(num_experts) + "_sorted_token_ids",
                                     h_sorted_token_ids, d_sorted_token_ids, 0, 0));
        EXPECT_TRUE(CheckResult<int>("moe_align_block_ep_" + std::to_string(expert_para_size) + "_es_" +
                                         std::to_string(num_experts) + "_total_tokens_post_pad",
                                     h_total_tokens_post_pad, d_total_tokens_post_pad, 0, 0));

        DeleteBuffer(d_cumsum);
      }

      // Free data
      DeleteBuffer(h_topk_ids);
      DeleteBuffer(h_sorted_token_ids);
      DeleteBuffer(h_expert_ids);
      DeleteBuffer(h_total_tokens_post_pad);
      DeleteBuffer(d_topk_ids);
      DeleteBuffer(d_sorted_token_ids);
      DeleteBuffer(d_expert_ids);
      DeleteBuffer(d_total_tokens_post_pad);
    }
  }
}

// Disable performance test by default
TEST_F(LlamaNvidiaMoeTestSuit, DISABLED_MoeAlignBlockKernelPerfTest) {
  const std::vector<int> token_nums = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  // Config of DeepSeek-R1
  const int topk = 8;
  const int block_size = 64;
  const int expert_para_size = 1;
  const int num_experts = 256;

  for (const int token_num : token_nums) {
    const int num_experts_per_rank = num_experts / expert_para_size;
    const size_t numel = token_num * topk;
    const int max_num_tokens_padded = numel + num_experts_per_rank * (block_size - 1);
    const int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;

    // Prepare device data
    BufferMeta d_topk_ids =
        CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(topk)},
                          /*is_random_init*/ true, /*min_val*/ 0, /*max_val*/ num_experts - 1);
    BufferMeta d_sorted_token_ids =
        CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(max_num_tokens_padded)},
                          /*is_random_init*/ false);
    BufferMeta d_expert_ids = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(max_num_m_blocks)},
                                                /*is_random_init*/ false);
    BufferMeta d_total_tokens_post_pad = CreateBuffer<int>(MemoryType::MEMORY_GPU, {1},
                                                           /*is_random_init*/ false);
    BufferMeta d_cumsum = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_experts + 1)},
                                            /*is_random_init*/ false);

    const int warmups = 5;
    const int iterations = 10;
    auto cuda_run = [&]() {
      InvokeSglMoeAlignBlockSize<int>(
          reinterpret_cast<int*>(d_topk_ids.data_ptr), reinterpret_cast<int*>(d_sorted_token_ids.data_ptr),
          reinterpret_cast<int*>(d_expert_ids.data_ptr), reinterpret_cast<int*>(d_total_tokens_post_pad.data_ptr),
          max_num_tokens_padded, num_experts, block_size, numel, reinterpret_cast<int*>(d_cumsum.data_ptr), stream);
    };
    float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Token num: " << token_num << ", Execution time: " << elapsed_ms << " ms" << std::endl;

    // Free data
    DeleteBuffer(d_topk_ids);
    DeleteBuffer(d_sorted_token_ids);
    DeleteBuffer(d_expert_ids);
    DeleteBuffer(d_total_tokens_post_pad);
    DeleteBuffer(d_cumsum);
  }
}

TEST_F(LlamaNvidiaMoeTestSuit, FillIntToBufferTest) {
  std::vector<int> fill_info;
  std::vector<int> test_fill_length = {32684, 7, 10};
  fill_info.insert(fill_info.end(), {0, test_fill_length[0], -1});
  fill_info.insert(fill_info.end(), {test_fill_length[0], test_fill_length[1], 0});
  fill_info.insert(fill_info.end(), {test_fill_length[0] + test_fill_length[1], test_fill_length[2], 1});
  size_t total_length = std::accumulate(test_fill_length.begin(), test_fill_length.end(), 0);

  void* output_ptr;
  void* fill_info_ptr;
  cudaMalloc(&output_ptr, total_length * sizeof(int));
  cudaMalloc(&fill_info_ptr, fill_info.size() * sizeof(int));
  InvokeFillIntToBuffer(static_cast<int*>(output_ptr), fill_info_ptr, fill_info.data(), fill_info.size(), stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<int> device_output(total_length);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(static_cast<void*>(device_output.data()), output_ptr, total_length * sizeof(int),
                                     cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < total_length; ++i) {
    if (i < static_cast<size_t>(test_fill_length[0])) {
      EXPECT_EQ(device_output[i], -1);
    } else if (i < static_cast<size_t>(test_fill_length[0] + test_fill_length[1])) {
      EXPECT_EQ(device_output[i], 0);
    } else {
      EXPECT_EQ(device_output[i], 1);
    }
  }

  cudaFree(output_ptr);
  cudaFree(fill_info_ptr);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
