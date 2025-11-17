/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "cuda_fp8_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"
namespace llm_kernels {
namespace nvidia {
namespace test {
class LLMKernelsNvidiaUtilsTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T_IN>
  void ComputeFP8QuantizeScalePrecisionTestImpl() {
    // <num_channels, channel_size>
    using testcase_t = std::pair<size_t, size_t>;
    std::vector<testcase_t> testcases = {{1, 31}, {1, 16383}, {1, 5120}, {1, 12800}, {1, 4096}, {2048, 5120}, {2048, 12800}, {2048, 4096}};
    for (testcase_t& shape : testcases) {
      int32_t num_channels = shape.first;
      int32_t channel_size = shape.second;

      BufferMeta input = this->CreateBuffer<T_IN>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, /*is_random_init*/ true);
      T_IN* input_ptr = reinterpret_cast<T_IN*>(input.data_ptr);

      BufferMeta input_host = this->CopyToHost<T_IN>(input);
      T_IN* input_host_ptr = reinterpret_cast<T_IN*>(input_host.data_ptr);

      BufferMeta output = this->CreateBuffer<float>(MemoryType::MEMORY_GPU, {shape.first}, /*is_random_init*/ false);
      float* output_ptr = reinterpret_cast<float*>(output.data_ptr);
      InvokeComputeFP8QuantizeScale(output_ptr, input_ptr, num_channels, channel_size, stream);

      BufferMeta output_host = this->CopyToHost<float>(output);
      float* output_host_ptr = static_cast<float*>(output_host.data_ptr);

      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      for (int n = 0; n < num_channels; ++n) {
        float channel_max = 0.f;
        for (int k = 0; k < channel_size; ++k) {
          float val = fabs(static_cast<float>(input_host_ptr[n * channel_size + k]));
          channel_max = std::max(val, channel_max);
        }
        channel_max = std::max(channel_max / FP8_E4M3_MAX, FP8_E4M3_MIN_SCALE);
        EXPECT_TRUE(this->AlmostEqual(channel_max, output_host_ptr[n], 1e-6));
      }
      std::cout << "completed the precision test of shape {" << num_channels << ", " << channel_size << "}" << std::endl;
      this->DeleteBuffer(input);
      this->DeleteBuffer(input_host);
      this->DeleteBuffer(output);
      this->DeleteBuffer(output_host);
    }
  }

  template <typename T_IN>
  void ComputeFP8QuantizeScalePerformanceTestImpl() {
    using testcase_t = std::pair<size_t, size_t>;
    std::vector<testcase_t> testcases;
    for (size_t i = 8; i < 128; i += 8) {
      testcases.push_back({i, 5120});
      testcases.push_back({i, 12800});
      testcases.push_back({i, 4096});
    }

    for (size_t i = 512; i < 4096; i += 512) {
      testcases.push_back({i, 5120});
      testcases.push_back({i, 12800});
      testcases.push_back({i, 4096});
    }

    for (testcase_t& shape : testcases) {
      int32_t num_channels = shape.first;
      int32_t channel_size = shape.second;

      BufferMeta input = this->CreateBuffer<T_IN>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, /*is_random_init*/ true);
      T_IN* input_ptr = reinterpret_cast<T_IN*>(input.data_ptr);

      BufferMeta output = this->CreateBuffer<float>(MemoryType::MEMORY_GPU, {shape.first}, /*is_random_init*/ false);
      float* output_ptr = reinterpret_cast<float*>(output.data_ptr);

      const int num_iterations = 50;
      auto cuda_run = [&]() {
        InvokeComputeFP8QuantizeScale(output_ptr, input_ptr, num_channels, channel_size, stream);
      };

      float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, num_iterations, num_iterations);
      std::cout << "InvokeComputeFP8QuantizeScale shape:{" << num_channels << ", " << channel_size << "} Average elapsed time: " << elapsed_ms << " ms" << std::endl;
      this->DeleteBuffer(input);
      this->DeleteBuffer(output);
    }
  }

  template <typename T_IN>
  void QuantizeMatrixPrecisionTestImpl() {
    // <num_channels, channel_size>
    using testcase_t = std::pair<size_t, size_t>;
    std::vector<testcase_t> testcases = {{1, 31}, {1, 16383}, {1, 5120}, {1, 12800}, {1, 4096}, {2048, 5120}, {2048, 12800}, {2048, 4096}};
    for (testcase_t& shape : testcases) {
      int32_t num_channels = shape.first;
      int32_t channel_size = shape.second;

      BufferMeta input = this->CreateBuffer<T_IN>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, true);
      T_IN* input_ptr = reinterpret_cast<T_IN*>(input.data_ptr);

      BufferMeta input_host = this->CopyToHost<T_IN>(input);
      T_IN* input_host_ptr = reinterpret_cast<T_IN*>(input_host.data_ptr);

      BufferMeta scale = this->CreateBuffer<float>(MemoryType::MEMORY_GPU, {shape.first}, true, FP8_E4M3_MIN_SCALE, 1.f);
      float* scale_ptr = reinterpret_cast<float*>(scale.data_ptr);

      BufferMeta scale_host = this->CopyToHost<float>(scale);
      float* scale_host_ptr = reinterpret_cast<float*>(scale_host.data_ptr);

      BufferMeta output =
          this->CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, /*is_random_init*/ false);
      __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr);

      InvokeQuantizeMatrix(output_ptr, scale_ptr, input_ptr, num_channels, channel_size, stream);

      BufferMeta output_host = this->CopyToHost<__nv_fp8_e4m3>(output);

      __nv_fp8_e4m3* output_host_ptr = static_cast<__nv_fp8_e4m3*>(output_host.data_ptr);

      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      for (int n = 0; n < num_channels; ++n) {
        for (int k = 0; k < channel_size; ++k) {
          float val = static_cast<float>(input_host_ptr[n * channel_size + k]);
          val = std::min(std::max(val / scale_host_ptr[n], -FP8_E4M3_MAX), FP8_E4M3_MAX);
          val = static_cast<float>(static_cast<__nv_fp8_e4m3>(val));
          EXPECT_TRUE(AlmostEqual(val, static_cast<float>(output_host_ptr[n * channel_size + k]), 1e-3, 1e-4));
        }
      }
      std::cout << "completed the precision test of shape {" << num_channels << ", " << channel_size << "}" << std::endl;
      this->DeleteBuffer(input);
      this->DeleteBuffer(input_host);
      this->DeleteBuffer(scale);
      this->DeleteBuffer(scale_host);
      this->DeleteBuffer(output);
      this->DeleteBuffer(output_host);
    }
  }

  template <typename T_IN>
  void QuantizeMatrixPerformanceTestImpl() {
    // <num_channels, channel_size>
    using testcase_t = std::pair<size_t, size_t>;
    std::vector<testcase_t> testcases;
    for (size_t i = 8; i < 128; i += 8) {
      testcases.push_back({i, 5120});
      testcases.push_back({i, 12800});
      testcases.push_back({i, 4096});
    }
    
    for (size_t i = 512; i < 4096; i += 512) {
      testcases.push_back({i, 5120});
      testcases.push_back({i, 12800});
      testcases.push_back({i, 4096});
    }

    for (testcase_t& shape : testcases) {
      int32_t num_channels = shape.first;
      int32_t channel_size = shape.second;

      BufferMeta input = this->CreateBuffer<T_IN>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, true);
      T_IN* input_ptr = reinterpret_cast<T_IN*>(input.data_ptr);

      BufferMeta scale = this->CreateBuffer<float>(MemoryType::MEMORY_GPU, {shape.first}, true, FP8_E4M3_MIN_SCALE, 1.f);
      float* scale_ptr = reinterpret_cast<float*>(scale.data_ptr);

      BufferMeta output =
          this->CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {shape.first, shape.second}, /*is_random_init*/ false);
      __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr);

      const int num_iterations = 50;
      auto cuda_run = [&](){
        InvokeQuantizeMatrix(output_ptr, scale_ptr, input_ptr, num_channels, channel_size, stream);
      };

      float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, num_iterations, num_iterations);
      std::cout << "InvokeQuantizeMatrix shape:{" << num_channels << ", " << channel_size << "} Average elapsed time: " << elapsed_ms << " ms" << std::endl;
      this->DeleteBuffer(input);
      this->DeleteBuffer(scale);
      this->DeleteBuffer(output);
    }
  }
};

#ifdef ENABLE_FP8
TEST_F(LLMKernelsNvidiaUtilsTestSuit, ComputeFP8QuantizeScalePrecisionTest_half) {
  this->ComputeFP8QuantizeScalePrecisionTestImpl<half>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, ComputeFP8QuantizeScalePrecisionTest_bfloat16) {
  this->ComputeFP8QuantizeScalePrecisionTestImpl<__nv_bfloat16>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, DISABLED_ComputeFP8QuantizeScalePerformanceTest_half) {
  this->ComputeFP8QuantizeScalePerformanceTestImpl<half>();
}


TEST_F(LLMKernelsNvidiaUtilsTestSuit, DISABLED_ComputeFP8QuantizeScalePerformanceTest_bfloat16) {
  this->ComputeFP8QuantizeScalePerformanceTestImpl<__nv_bfloat16>();
}


TEST_F(LLMKernelsNvidiaUtilsTestSuit, QuantizeMatrixPrecisionTest_half) {
  this->QuantizeMatrixPrecisionTestImpl<half>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, QuantizeMatrixPrecisionTest_bfloat16) {
  this->QuantizeMatrixPrecisionTestImpl<__nv_bfloat16>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, DISABLED_QuantizeMatrixPerformanceTest_half) {
  this->QuantizeMatrixPerformanceTestImpl<half>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, DISABLED_QuantizeMatrixPerformanceTest_bfloat16) {
  this->QuantizeMatrixPerformanceTestImpl<__nv_bfloat16>();
}

TEST_F(LLMKernelsNvidiaUtilsTestSuit, ReScaleFp8E4m3Test) {
  // <num_channels, channel_size>
  std::vector<size_t> testcases = {1, 7, 31, 16383};
  for (size_t num_elems : testcases) {
    BufferMeta input = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_elems}, true);
    __nv_fp8_e4m3* input_ptr = reinterpret_cast<__nv_fp8_e4m3*>(input.data_ptr);

    BufferMeta input_host = CopyToHost<__nv_fp8_e4m3>(input);
    __nv_fp8_e4m3* input_host_ptr = reinterpret_cast<__nv_fp8_e4m3*>(input_host.data_ptr);

    BufferMeta input_scale = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1}, true, FP8_E4M3_MIN_SCALE, 1.f);
    float* input_scale_ptr = reinterpret_cast<float*>(input_scale.data_ptr);

    BufferMeta input_scale_host = CopyToHost<float>(input_scale);
    float* input_scale_host_ptr = reinterpret_cast<float*>(input_scale_host.data_ptr);

    BufferMeta output_scale = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1}, true, FP8_E4M3_MIN_SCALE, 1.f);
    float* output_scale_ptr = reinterpret_cast<float*>(output_scale.data_ptr);

    BufferMeta output_scale_host = CopyToHost<float>(output_scale);
    float* output_scale_host_ptr = reinterpret_cast<float*>(output_scale_host.data_ptr);

    BufferMeta output = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_elems}, /*is_random_init*/ false);
    __nv_fp8_e4m3* output_ptr = reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr);

    InvokeRescaleFp8E4m3(input_ptr, output_ptr, (int)num_elems, input_scale_ptr, output_scale_ptr, stream);

    BufferMeta output_host = CopyToHost<__nv_fp8_e4m3>(output);
    __nv_fp8_e4m3* output_host_ptr = static_cast<__nv_fp8_e4m3*>(output_host.data_ptr);

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (size_t n = 0; n < num_elems; ++n) {
      float val = static_cast<float>(input_host_ptr[n]);
      val = std::min(std::max(val * input_scale_host_ptr[0] / output_scale_host_ptr[0], -FP8_E4M3_MAX), FP8_E4M3_MAX);
      val = static_cast<float>(static_cast<__nv_fp8_e4m3>(val));
      EXPECT_TRUE(AlmostEqual(val, static_cast<float>(output_host_ptr[n]), 1e-3, 1e-4));
    }
  }
}
#endif  // ENABLE_FP8

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
