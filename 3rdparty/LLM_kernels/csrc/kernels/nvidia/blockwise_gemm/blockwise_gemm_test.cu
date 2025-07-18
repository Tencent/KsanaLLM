/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "blockwise_gemm.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include <cuda_fp8.h>

namespace llm_kernels {
namespace nvidia {
namespace test {

class BlockwiseGemmTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

__global__ void ConvertToFp8Kernel(const half* input, __nv_fp8_e4m3* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = __nv_fp8_e4m3(input[idx]);
  }
}

__global__ void ConvertFromFp8Kernel(const __nv_fp8_e4m3* input, half* output, size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = half(float(input[idx]));
  }
}

void ConvertToFp8(const half* input, __nv_fp8_e4m3* output, size_t num_elements, cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
  ConvertToFp8Kernel<<<num_blocks, threads_per_block, 0, stream>>>(input, output, num_elements);
}

void ConvertFromFp8(const __nv_fp8_e4m3* input, half* output, size_t num_elements, cudaStream_t stream) {
  int threads_per_block = 256;
  int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
  ConvertFromFp8Kernel<<<num_blocks, threads_per_block, 0, stream>>>(input, output, num_elements);
}

template <typename T>
void PrintData(const std::vector<T>& data, const std::string& name, int limit = 10) {
  std::cout << "=== " << name << " (showing first " << limit << " elements) ===" << std::endl;
  for (int i = 0; i < std::min(limit, static_cast<int>(data.size())); i++) {
    std::cout << "idx " << i << " = " << std::fixed << std::setprecision(6) << static_cast<float>(data[i]) << std::endl;
  }
}

template <typename T>
void PrintCast(std::vector<char>& data) {
  T* data_ptr = reinterpret_cast<T*>(data.data());
  for (int i = 0; i < 32; i++) {
    std::cout << "idx " << i << " = " << std::fixed << std::setprecision(6) << static_cast<float>(data_ptr[i])
              << std::endl;
  }
}

template <typename T>
std::vector<char> ReadTensorData(const std::string& data_file, size_t num_elements) {
  std::ifstream file(data_file, std::ios::binary);
  std::vector<char> data(num_elements * sizeof(T));
  file.read(data.data(), num_elements * sizeof(T));
  std::cout << "=====================" << data_file << "=================" << std::endl;
  PrintCast<T>(data);
  return data;
}

void* CopyToGPU(std::vector<char>& data) {
  void* d_data;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_data, data.size()));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice));
  return d_data;
}

void* ToFp8(std::vector<char>& data) {
  void* d_data;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_data, data.size() / 2));
  void* tmp_data = CopyToGPU(data);
  ConvertToFp8(reinterpret_cast<const half*>(tmp_data), static_cast<__nv_fp8_e4m3*>(d_data), data.size() / 2, nullptr);
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(tmp_data));
  return d_data;
}

float QuantizeToFp8(float value, float scale) {
  const float max_fp8 = 448.0f;
  float scaled_value = value / scale;
  scaled_value = std::max(std::min(scaled_value, max_fp8), -max_fp8);
  const float fp8_precision = 1.0f / 32.0f;
  scaled_value = std::round(scaled_value / fp8_precision) * fp8_precision;
  return scaled_value * scale;
}

float SimulateFp8Conversion(float value) {
  const float max_fp8 = 448.0f;
  const float min_fp8 = 1.0f / 16.0f;

  if (std::abs(value) < min_fp8) {
    return 0.0f;
  } else if (std::abs(value) > max_fp8) {
    return value > 0 ? max_fp8 : -max_fp8;
  } else {
    float scale = std::pow(2.0f, std::floor(std::log2(std::abs(value))));
    float normalized = value / scale;
    float quantized = std::round(normalized * 8.0f) / 8.0f;
    return quantized * scale;
  }
}

void ReferenceGemm(const std::vector<__nv_fp8_e4m3>& a, const std::vector<float>& a_scales,
                   const std::vector<__nv_fp8_e4m3>& b, const std::vector<float>& b_scales, std::vector<half>& c, int m,
                   int k, int n) {
  const int block_size_n = 128;
  const int block_size_k = 128;

  const int n_tiles = (n + block_size_n - 1) / block_size_n;
  const int k_tiles = (k + block_size_k - 1) / block_size_k;

  std::vector<float> b_transposed(k * n);
  for (int j = 0; j < n; ++j) {
    for (int l = 0; l < k; ++l) {
      b_transposed[l * n + j] = float(b[j * k + l]);
    }
  }

  std::fill(c.begin(), c.end(), half(0.0f));

  for (int i = 0; i < k_tiles; ++i) {
    for (int j = 0; j < n_tiles; ++j) {
      int k_start = i * block_size_k;
      int k_end = std::min((i + 1) * block_size_k, k);
      int n_start = j * block_size_n;
      int n_end = std::min((j + 1) * block_size_n, n);

      float scale = a_scales[i] * b_scales[j * k_tiles + i];
      for (int row = 0; row < m; ++row) {
        for (int col = n_start; col < n_end; ++col) {
          float sum = 0.0f;
          for (int l = k_start; l < k_end; ++l) {
            float a_val = SimulateFp8Conversion(float(a[row * k + l]));
            float b_val = SimulateFp8Conversion(b_transposed[l * n + col]);
            sum += a_val * b_val;
          }
          c[row * n + col] = static_cast<half>(static_cast<float>(c[row * n + col]) + sum * scale);
        }
      }
    }
  }
}

template <typename T>
std::vector<T> GenerateRandomData(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
  std::vector<T> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);

  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(dist(gen));
  }

  return data;
}

float CalcRelativeError(const std::vector<half>& a, const std::vector<half>& b) {
  float max_error = 0.0f;
  float max_val = 0.0f;

  for (size_t i = 0; i < a.size(); ++i) {
    float a_val = static_cast<float>(a[i]);
    float b_val = static_cast<float>(b[i]);
    float abs_error = std::abs(a_val - b_val);
    max_error = std::max(max_error, abs_error);
    max_val = std::max(max_val, std::abs(a_val));
  }

  return max_val > 0.0f ? max_error / max_val : max_error;
}

TEST_F(BlockwiseGemmTestSuit, BlockwiseKernelTestLegacy) {
  int major = 0, minor = 0;
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  std::vector<char> a = ReadTensorData<half>("/local_data/zezhao/work/data/A.bin", 2048 * 18432);
  std::vector<char> a_scale = ReadTensorData<float>("/local_data/zezhao/work/data/ScaleA.bin", 2048 * 144);
  std::vector<char> b = ReadTensorData<half>("/local_data/zezhao/work/data/B.bin", 18432 * 7168);
  std::vector<char> b_scale = ReadTensorData<float>("/local_data/zezhao/work/data/ScaleB.bin", 144 * 56);
  std::vector<char> c = ReadTensorData<half>("/local_data/zezhao/work/data/C.bin", 2048 * 7168);
  std::vector<char> out(2048 * 7168 * 2);

  void* d_a = ToFp8(a);
  void* d_a_scale = CopyToGPU(a_scale);
  void* d_b = ToFp8(b);
  void* d_b_scale = CopyToGPU(b_scale);
  void* d_out = CopyToGPU(out);

  cudaStream_t stream;
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));
  BlockwiseGemmKernel<half>(static_cast<void*>(d_a), static_cast<float*>(d_a_scale), static_cast<void*>(d_b),
                            static_cast<float*>(d_b_scale), static_cast<void*>(d_out), 2048, 18432, 7168, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));

  std::cout << "====================" << std::endl;
  PrintCast<half>(out);

  // Free GPU memory
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_a));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_a_scale));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_b));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_b_scale));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(d_out));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
}

TEST_F(BlockwiseGemmTestSuit, BlockwiseGemmPrecisionTest) {
  int major = 0, minor = 0;
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  const int m = 128;
  const int k = 256;
  const int n = 64;

  const int block_size_n = 128;
  const int block_size_k = 128;

  const int num_blocks_n = (n + block_size_n - 1) / block_size_n;
  const int num_blocks_k = (k + block_size_k - 1) / block_size_k;

  std::vector<half> a_half_host = GenerateRandomData<half>(m * k);
  std::vector<half> b_half_host = GenerateRandomData<half>(k * n);

  std::vector<float> a_scales_host = GenerateRandomData<float>(num_blocks_k, 0.5f, 2.0f);
  std::vector<float> b_scales_host = GenerateRandomData<float>(num_blocks_n * num_blocks_k, 0.5f, 2.0f);

  std::vector<half> c_host(m * n, half(0.0f));
  std::vector<half> c_ref_host(m * n, half(0.0f));

  std::vector<__nv_fp8_e4m3> a_fp8_host(m * k);
  std::vector<__nv_fp8_e4m3> b_fp8_host(k * n);

  for (size_t i = 0; i < a_half_host.size(); ++i) {
    a_fp8_host[i] = __nv_fp8_e4m3(float(a_half_host[i]));
  }

  for (size_t i = 0; i < b_half_host.size(); ++i) {
    b_fp8_host[i] = __nv_fp8_e4m3(float(b_half_host[i]));
  }

  ReferenceGemm(a_fp8_host, a_scales_host, b_fp8_host, b_scales_host, c_ref_host, m, k, n);

  half* a_device;
  half* b_device;
  half* c_device;
  float* a_scales_device;
  float* b_scales_device;
  __nv_fp8_e4m3* a_fp8_device;
  __nv_fp8_e4m3* b_fp8_device;

  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_device, m * k * sizeof(half)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_device, k * n * sizeof(half)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&c_device, m * n * sizeof(half)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_scales_device, num_blocks_k * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_scales_device, num_blocks_n * num_blocks_k * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_fp8_device, m * k * sizeof(__nv_fp8_e4m3)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_fp8_device, k * n * sizeof(__nv_fp8_e4m3)));

  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(a_device, a_half_host.data(), m * k * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(b_device, b_half_host.data(), k * n * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpy(a_scales_device, a_scales_host.data(), num_blocks_k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(b_scales_device, b_scales_host.data(), num_blocks_n * num_blocks_k * sizeof(float),
                                     cudaMemcpyHostToDevice));

  ConvertToFp8(a_device, a_fp8_device, m * k, stream);
  ConvertToFp8(b_device, b_fp8_device, k * n, stream);

  BlockwiseGemmKernel<half>(static_cast<void*>(a_fp8_device), a_scales_device, static_cast<void*>(b_fp8_device),
                            b_scales_device, static_cast<void*>(c_device), m, k, n, stream);

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(c_host.data(), c_device, m * n * sizeof(half), cudaMemcpyDeviceToHost));

  PrintData(c_host, "BlockwiseGemm Result");
  PrintData(c_ref_host, "Reference Result");

  float rel_error = CalcRelativeError(c_host, c_ref_host);
  std::cout << "Relative error: " << rel_error << std::endl;

  EXPECT_LT(rel_error, 2.0f);

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(c_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_scales_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_scales_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_fp8_device));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_fp8_device));
}

// Performance test
TEST_F(BlockwiseGemmTestSuit, BlockwiseGemmPerformanceTest) {
  int major = 0, minor = 0;
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 9) {
    GTEST_SKIP() << "Skipping test because SM version is less than 90";
    return;
  }

  std::vector<std::tuple<int, int, int>> test_sizes = {
      {128, 128, 128}, {128, 512, 512}, {128, 1024, 1024}, {128, 10240, 2560}};

  for (const auto& size : test_sizes) {
    int m = std::get<0>(size);
    int k = std::get<1>(size);
    int n = std::get<2>(size);

    const int block_size_n = 128;
    const int block_size_k = 128;

    const int num_blocks_n = (n + block_size_n - 1) / block_size_n;
    const int num_blocks_k = (k + block_size_k - 1) / block_size_k;

    std::vector<half> a_half_host = GenerateRandomData<half>(m * k);
    std::vector<half> b_half_host = GenerateRandomData<half>(k * n);

    std::vector<float> a_scales_host = GenerateRandomData<float>(num_blocks_k, 0.5f, 2.0f);
    std::vector<float> b_scales_host = GenerateRandomData<float>(num_blocks_n * num_blocks_k, 0.5f, 2.0f);

    std::vector<half> c_host(m * n, half(0.0f));

    half* a_device;
    half* b_device;
    half* c_device;
    float* a_scales_device;
    float* b_scales_device;
    __nv_fp8_e4m3* a_fp8_device;
    __nv_fp8_e4m3* b_fp8_device;
    void* cutlass_buffer = nullptr;
    size_t cutlass_buffer_size = m * k * sizeof(float);

    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_device, m * k * sizeof(half)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_device, k * n * sizeof(half)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&c_device, m * n * sizeof(half)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_scales_device, num_blocks_k * sizeof(float)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_scales_device, num_blocks_n * num_blocks_k * sizeof(float)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&a_fp8_device, m * k * sizeof(__nv_fp8_e4m3)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&b_fp8_device, k * n * sizeof(__nv_fp8_e4m3)));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&cutlass_buffer, m * k * sizeof(float)));

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(a_device, a_half_host.data(), m * k * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(b_device, b_half_host.data(), k * n * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(a_scales_device, a_scales_host.data(), num_blocks_k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(b_scales_device, b_scales_host.data(),
                                       num_blocks_n * num_blocks_k * sizeof(float), cudaMemcpyHostToDevice));

    ConvertToFp8(a_device, a_fp8_device, m * k, stream);
    ConvertToFp8(b_device, b_fp8_device, k * n, stream);

    for (int i = 0; i < 5; ++i) {
      BlockwiseGemmKernel<half>(static_cast<void*>(a_fp8_device), a_scales_device, static_cast<void*>(b_fp8_device),
                                b_scales_device, static_cast<void*>(c_device), m, k, n, stream, cutlass_buffer,
                                cutlass_buffer_size);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    const int num_iterations = 10;
    auto cuda_run = [&]() {
      BlockwiseGemmKernel<half>(static_cast<void*>(a_fp8_device), a_scales_device, static_cast<void*>(b_fp8_device),
                                b_scales_device, static_cast<void*>(c_device), m, k, n, stream, cutlass_buffer,
                                cutlass_buffer_size);
    };
    float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, num_iterations, num_iterations);

    double flops = 2.0 * m * n * k;
    double gflops = (flops * 1e-9) / (elapsed_ms * 1e-3);

    std::cout << "Matrix size: " << m << "x" << k << "x" << n << ", Execution time: " << elapsed_ms << " ms"
              << ", Performance: " << gflops << " GFLOPS" << std::endl;

    CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(c_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_scales_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_scales_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(a_fp8_device));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(b_fp8_device));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
