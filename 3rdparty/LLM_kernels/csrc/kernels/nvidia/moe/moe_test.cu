/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "moe.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

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
    InvokeMoeSum<float, false>(d_input, d_output, nullptr, nullptr, num_tokens, num_experts, topk, hidden_size, stream);

  cudaEvent_t start, stop;
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));

  /*
  for (int k = 1; k <= 80; k++) {
    if (k > 16 && k % 8 != 0) continue;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start, stream));

    for (int i = 0; i < 1000; ++i) {
      InvokeMoeSum<float, false>(
          d_input, d_output, nullptr, nullptr, num_tokens, num_experts, topk, hidden_size, stream);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution 1 times(TopK = " << k << "): " << (milliseconds / 1000) << " ms" << std::endl;
  }
  */

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
                         nullptr, nullptr, 256, num_elements, inter_size, stream);
    FlashinferSiluAndMul<T>(reinterpret_cast<const T*>(d_input.data_ptr),
                            reinterpret_cast<T*>(d_output_flashinfer.data_ptr), nullptr, nullptr, 256, num_elements,
                            inter_size, stream);

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
                           nullptr, nullptr, 256, num_elements, inter_size, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << std::setw(25) << "SiluAndMul "
              << "Kernel execution 1 times " << std::setw(10) << milliseconds << " ms" << std::endl;

    auto cuda_run_flashinfer = [&]() {
      FlashinferSiluAndMul<T>(reinterpret_cast<const T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output.data_ptr),
                              nullptr, nullptr, 256, num_elements, inter_size, stream);
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

void MoeAlignBlockCpu(std::vector<int>& topk_ids, std::vector<int>& expert_ids, std::vector<int>& sorted_ids,
                      std::vector<int>& expert_map, std::vector<int>& token_post_pad, int token_num, int topk,
                      int expert_num, int block_size) {
  std::vector<int> cumsum(expert_num + 1);
  std::vector<int> token_cnts(expert_num + 1);
  size_t numel = static_cast<size_t>(token_num) * topk;
  for (size_t i = 0; i < numel; ++i) {
    int expert_id = expert_map[topk_ids[i]];
    if (expert_id >= expert_num) {
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
    int expert_id = expert_map[topk_ids[i]];
    if (expert_id >= expert_num) {
      continue;
    }
    int idx = cumsum[expert_id] * block_size + token_cnts[expert_id];
    sorted_ids[idx] = i;
    token_cnts[expert_id] += 1;
  }
}

TEST_F(LlamaNvidiaMoeTestSuit, MoeAlignBlockKernelTest) {
  int token_num = 4;
  int expert_para_size = 2;
  int topk = 6;
  int block_size = 64;
  int num_thread = 256;

  // 测试不同num_experts下的性能
  std::vector<int> expert_sizes = {8, 16, 32, 64, 128, 256};

  for (int num_experts : expert_sizes) {
    std::cout << "\n===== Testing with num_experts = " << num_experts << " =====" << std::endl;

    int num_experts_per_rank = num_experts / expert_para_size;
    size_t numel = token_num * topk;
    int max_num_tokens_padded = numel + num_experts_per_rank * (block_size - 1);
    int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    int shared_mem = ((num_thread + 2) * num_experts) * sizeof(uint16_t) + (num_experts + 1) * sizeof(int32_t);
    std::cout << "Shared Mem = " << (shared_mem / 1024) << "KB" << std::endl;

    int device_max_shared_mem;
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&device_max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (device_max_shared_mem <= shared_mem) {
      std::cout << "Current GPU Device do not support Shared Memory " << shared_mem
                << ", cudaDevAttrMaxSharedMemoryPerBlockOptin = " << device_max_shared_mem << std::endl;
      continue;
    }

    std::vector<int> topk_ids(token_num * topk);
    // 生成随机的0到num_experts-1之间的整数填充topk_ids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, num_experts - 1);
    for (size_t i = 0; i < token_num * topk; ++i) {
      topk_ids[i] = dist(gen);
    }

    std::vector<int> expert_map(num_experts);
    for (int i = 0; i < num_experts; i++) {
      if (expert_para_size == 1) {
        expert_map[i] = i;
      } else {
        if (i < num_experts_per_rank) {
          expert_map[i] = num_experts_per_rank + 1;
        } else {
          expert_map[i] = i - num_experts_per_rank;
        }
      }
    }

    // 使用 CPU 计算理论的 MoeAlignBlock 输出
    std::vector<int> sorted_ids(max_num_tokens_padded, num_experts_per_rank);
    std::vector<int> expert_ids(max_num_m_blocks, -1);
    std::vector<int> token_post_pad(1, -1);
    MoeAlignBlockCpu(topk_ids, expert_ids, sorted_ids, expert_map, token_post_pad, token_num, topk,
                     num_experts_per_rank, block_size);

    // Cuda 计算结果
    std::vector<int> h_sorted_ids(max_num_tokens_padded, num_experts_per_rank);
    std::vector<int> h_expert_ids(max_num_m_blocks, -1);
    std::vector<int> h_token_post_pad(1, -1);

    void* d_topk_ids;
    void* d_sorted_token_ids;
    void* d_experts_ids;
    void* d_total_tokens_post_pad;
    void* d_expert_map;
    cudaMalloc(&d_topk_ids, token_num * topk * sizeof(int));
    cudaMalloc(&d_sorted_token_ids, max_num_tokens_padded * sizeof(int));
    cudaMalloc(&d_experts_ids, max_num_m_blocks * sizeof(int));
    cudaMalloc(&d_total_tokens_post_pad, 1 * sizeof(int));
    cudaMalloc(&d_expert_map, num_experts * sizeof(int));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(d_topk_ids, reinterpret_cast<void*>(topk_ids.data()), numel * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_expert_map, reinterpret_cast<void*>(expert_map.data()),
                                       num_experts * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_sorted_token_ids, reinterpret_cast<void*>(h_sorted_ids.data()),
                                       max_num_tokens_padded * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_experts_ids, reinterpret_cast<void*>(h_expert_ids.data()),
                                       max_num_m_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_total_tokens_post_pad, reinterpret_cast<void*>(h_token_post_pad.data()),
                                       1 * sizeof(int), cudaMemcpyHostToDevice));

    // 预热运行
    for (int i = 0; i < 0; ++i) {
      InvokeMoeAlignBlockSize<int, uint16_t, true>(
          reinterpret_cast<int*>(d_topk_ids), reinterpret_cast<int*>(d_sorted_token_ids),
          reinterpret_cast<int*>(d_experts_ids), reinterpret_cast<int*>(d_total_tokens_post_pad),
          reinterpret_cast<int*>(d_expert_map), topk, num_experts_per_rank, expert_para_size, block_size, numel,
          0, stream);
    }

    cudaEvent_t start, stop;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));

    // 计时运行100次取平均值
    const int num_iterations = 1;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start, stream));

    for (int i = 0; i < num_iterations; ++i) {
      InvokeMoeAlignBlockSize<int, uint16_t, true>(
          reinterpret_cast<int*>(d_topk_ids), reinterpret_cast<int*>(d_sorted_token_ids),
          reinterpret_cast<int*>(d_experts_ids), reinterpret_cast<int*>(d_total_tokens_post_pad),
          reinterpret_cast<int*>(d_expert_map), topk, num_experts_per_rank, expert_para_size, block_size, numel, 0,
          stream);
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution " << num_iterations << " times, average: " << (milliseconds / num_iterations)
              << " ms" << std::endl;

    // 对比正确性结果 Expert Ids
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_expert_ids.data(), d_experts_ids, max_num_m_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < max_num_m_blocks; ++i) {
      EXPECT_EQ(h_expert_ids[i], expert_ids[i]);
    }
    // Sorted Ids
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_sorted_ids.data(), d_sorted_token_ids, max_num_tokens_padded * sizeof(int),
                                       cudaMemcpyDeviceToHost));
    for (int i = 0; i < max_num_tokens_padded; ++i) {
      EXPECT_EQ(h_sorted_ids[i], sorted_ids[i]);
    }

    // Token Post Pad
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(h_token_post_pad.data(), d_total_tokens_post_pad, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 1; ++i) {
      EXPECT_EQ(h_token_post_pad[i], token_post_pad[i]);
    }

    cudaFree(d_topk_ids);
    cudaFree(d_sorted_token_ids);
    cudaFree(d_experts_ids);
    cudaFree(d_total_tokens_post_pad);
    cudaFree(d_expert_map);
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventDestroy(stop));
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
    if (i < test_fill_length[0]) {
      EXPECT_EQ(device_output[i], -1);
    } else if (i < test_fill_length[0] + test_fill_length[1]) {
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
