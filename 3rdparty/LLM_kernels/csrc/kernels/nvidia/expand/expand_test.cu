/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/script.h>
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/expand/expand.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaExpandTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::device;
  using NvidiaTestSuitBase::stream;

  template <typename T>
  void TestExpand(const size_t m, const size_t n, const size_t expand_size) {
    std::string type_str = "float";
    torch::ScalarType tensor_type = torch::kFloat32;
    if (std::is_same<T, half>::value) {
      type_str = "half";
      tensor_type = torch::kFloat16;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      tensor_type = torch::kBFloat16;
    }
    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ true);
    BufferMeta expand_output_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, expand_size, n}, /*is_random_init*/ false);
    BufferMeta expand_output_torch_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, expand_size, n}, /*is_random_init*/ false);

    int num_iterations = 3;
    auto options = torch::TensorOptions().device(torch::kCUDA, device).dtype(tensor_type);
    torch::Tensor input_tensor =
        torch::from_blob(input_meta.data_ptr, {static_cast<int64_t>(m), 1, static_cast<int64_t>(n)}, options);

    cudaStream_t old_stream = torch::cuda::getCurrentCUDAStream(device).stream();

    torch::cuda::CUDAStream new_stream = torch::cuda::getStreamFromExternal(stream, device);
    torch::cuda::setCurrentCUDAStream(new_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
    for (int i = 0; i < 1; i++) {
      InvokeExpand<T>(reinterpret_cast<const T*>(input_meta.data_ptr),
                      reinterpret_cast<T*>(expand_output_meta.data_ptr), m, expand_size, n, 0, stream);

      torch::Tensor output_tensor_expanded =
          input_tensor.expand({static_cast<int64_t>(m), static_cast<int64_t>(expand_size), static_cast<int64_t>(n)})
              .contiguous();
      if (i == 0) {
        CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(expand_output_torch_meta.data_ptr, output_tensor_expanded.data_ptr(),
                                           sizeof(T) * m * n * expand_size, cudaMemcpyDeviceToDevice));
        CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
        EXPECT_TRUE(CheckResult<T>("expand_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                                   expand_output_torch_meta, expand_output_meta, 0.0f, 0.0f));
      }
    }
    cudaStreamSynchronize(stream);
    // Start test custom expand op
    float custom_time_ms = 0.0f;
    cudaEventRecord(start, stream);

    for (int i = 0; i < num_iterations; i++) {
      InvokeExpand<T>(reinterpret_cast<const T*>(input_meta.data_ptr),
                      reinterpret_cast<T*>(expand_output_meta.data_ptr), m, expand_size, n, 0, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&custom_time_ms, start, stop);
    custom_time_ms /= num_iterations;

    float torch_time_ms = 0.0f;
    cudaEventRecord(start, stream);

    for (int i = 0; i < num_iterations; i++) {
      torch::Tensor output_tensor_expanded =
          input_tensor.expand({static_cast<int64_t>(m), static_cast<int64_t>(expand_size), static_cast<int64_t>(n)})
              .contiguous();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&torch_time_ms, start, stop);
    torch_time_ms /= num_iterations;

    // 输出结果
    std::cout << "Input shape: [" << m << ", 1, " << n << "], Expand size: " << expand_size << std::endl;
    std::cout << "Custom InvokeExpand time: " << custom_time_ms << " ms" << std::endl;
    std::cout << "Torch expand time: " << torch_time_ms << " ms" << std::endl;
    float speed_ratio = torch_time_ms / custom_time_ms;
    std::cout << "Speed ratio (Custom/Torch): " << speed_ratio << std::endl;
    int arch = GetSMVersion();
    if (arch == 90) {
      EXPECT_GE(speed_ratio, 3.0f);
    }

    torch::cuda::CUDAStream original_stream = torch::cuda::getStreamFromExternal(old_stream, device);
    torch::cuda::setCurrentCUDAStream(original_stream);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    DeleteBuffer(input_meta);
    DeleteBuffer(expand_output_meta);
    DeleteBuffer(expand_output_torch_meta);
  }
};

TEST_F(LlamaNvidiaExpandTestSuit, ExpandTest) { 
  TestExpand<half>(2048, 64, 128); 
  TestExpand<__nv_bfloat16>(2048, 64, 128);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
