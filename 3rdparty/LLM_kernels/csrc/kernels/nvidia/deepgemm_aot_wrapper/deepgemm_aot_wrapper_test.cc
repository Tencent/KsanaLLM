/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * This file contains test cases for the DeepGEMM AOT (Ahead-Of-Time) wrapper.
 * It tests the functionality of DeepGEMM matrix multiplication operations
 * with FP8 data types and scaling factors.
 */
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>

#include "csrc/kernels/nvidia/deepgemm_aot_wrapper/deepgemm_aot_wrapper.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class DeepGEMMAOTWrapperTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  /**
   * @brief Calculate the maximum absolute value for each token and scale to FP8 format
   *
   * This function processes each token (row) in the input tensor, finds the maximum
   * absolute value, and scales the values to fit within the FP8 range.
   *
   * @param x Input tensor of shape [m, n] where n must be divisible by 128
   * @return A tuple containing the scaled FP8 tensor and the scaling factors
   */
  std::tuple<torch::Tensor, torch::Tensor> per_token_cast_to_fp8(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 128 == 0,
                "Input tensor must be 2D and have second dimension divisible by 128");

    auto m = x.size(0);
    auto n = x.size(1);

    // Reshape x to (m, -1, 128) for token-wise processing
    auto x_view = x.view({m, -1, 128});

    // Calculate the maximum absolute value for each token
    auto x_amax = x_view.abs().to(torch::kFloat32).amax(2).view({m, -1}).clamp(1e-4);

    // Scale values and convert to float8 (e4m3fn format)
    auto scaled_x = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch::kFloat8_e4m3fn).view({m, n});
    auto scales = (x_amax / 448.0).view({m, -1});

    return {scaled_x, scales};
  }

  /**
   * @brief Calculate the maximum absolute value for each block and scale to FP8 format
   *
   * This function processes the input tensor in blocks of 128x128, finds the maximum
   * absolute value in each block, and scales the values to fit within the FP8 range.
   * It also handles padding to ensure the tensor dimensions are multiples of 128.
   *
   * @param x Input 2D tensor of any shape
   * @return A tuple containing the scaled FP8 tensor and the scaling factors
   */
  std::tuple<torch::Tensor, torch::Tensor> per_block_cast_to_fp8(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");

    auto m = x.size(0);
    auto n = x.size(1);

    // Calculate padded dimensions to ensure they are multiples of 128
    auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };
    auto padded_m = ceil_div(m, 128) * 128;
    auto padded_n = ceil_div(n, 128) * 128;

    // Create padded tensor filled with zeros
    auto x_padded = torch::zeros({padded_m, padded_n}, x.options()).contiguous();
    x_padded.slice(0, 0, m).slice(1, 0, n).copy_(x);

    // Reshape padded tensor into blocks of 128x128
    auto x_view = x_padded.view({-1, 128, padded_n / 128, 128});

    // Calculate maximum absolute value for each 128x128 block
    auto x_amax = x_view.abs().to(torch::kFloat32).amax({1, 3}, true).clamp(1e-4);

    // Scale values and convert to float8 (e4m3fn format)
    auto x_scaled = (x_view * (448.0 / x_amax)).to(torch::kFloat8_e4m3fn);

    // Restore original shape and extract valid portion (removing padding)
    auto result = x_scaled.view_as(x_padded).slice(0, 0, m).slice(1, 0, n).contiguous();
    auto scales = (x_amax / 448.0).view({x_view.size(0), x_view.size(2)});

    return {result, scales};
  }

  /**
   * @brief Get column-major TMA aligned tensor
   *
   * This function converts a tensor to column-major format for TMA (Tensor Memory Access)
   * alignment requirements. This is a simplified implementation.
   *
   * @param x Input tensor
   * @return Column-major aligned tensor
   */
  torch::Tensor get_col_major_tma_aligned_tensor(const torch::Tensor& x) {
    // Simplified implementation, actual implementation might require more complex logic
    return x.transpose(0, 1).contiguous();
  }

  /**
   * @brief Construct test data for DeepGEMM AOT wrapper testing
   *
   * This function creates random input tensors, prepares output tensors,
   * computes reference output using PyTorch's matmul, and converts inputs
   * to FP8 format with appropriate scaling factors. Equivalent to the Python
   * version of the construct function.
   *
   * @param m Number of rows in first input matrix
   * @param k Common dimension between input matrices
   * @param n Number of columns in second input matrix
   * @return Tuple containing FP8 inputs with scales, output tensor, and reference output
   */
  std::tuple<std::tuple<torch::Tensor, torch::Tensor>, std::tuple<torch::Tensor, torch::Tensor>, torch::Tensor,
             torch::Tensor>
  construct(int m, int k, int n) {
    // Create random input tensors
    auto x = torch::randn({m, k}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    auto y = torch::randn({n, k}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

    // Create output tensor
    auto out = torch::empty({m, n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

    // Compute reference output using PyTorch's matmul
    auto ref_out = torch::matmul(x, y.transpose(0, 1));

    // Convert to FP8 format with appropriate scaling
    auto [x_fp8, x_scales] = per_token_cast_to_fp8(x);
    auto [y_fp8, y_scales] = per_block_cast_to_fp8(y);

    // Transpose scales to match Python implementation
    auto x_scales_aligned = get_col_major_tma_aligned_tensor(x_scales);

    return {{x_fp8, x_scales_aligned}, {y_fp8, y_scales}, out, ref_out};
  }

  /**
   * @brief Test function for DeepGEMM AOT wrapper
   *
   * This function tests the DeepGEMM AOT wrapper by constructing test data,
   * running the matrix multiplication operation, and comparing the results
   * with a reference implementation.
   */
  void TestDeepGEMMAOTWrapper() {
    // Test parameters for matrix dimensions
    uint32_t m = 16;
    uint32_t k = 7168;
    uint32_t n = 36864;
    uint32_t smem_size = 158752;
    uint32_t num_sms = 78;

    // Execute gemm_algo_config_generator.py script
    std::cout << "Generating GEMM algorithm configuration..." << std::endl;
    DeepGEMMAOTWrapper deepgemm_aot_wrapper(m, n, k, /*need_generate_kernel*/ true,
                                            /*tuner_device_id*/ 0);

    std::vector<uint32_t> compute_ms({4, 8, 12, 16});
    for (auto compute_m : compute_ms) {
      // Construct test data including inputs and reference output
      auto result = construct(compute_m, k, n);

      // Extract elements from the nested tuple
      auto x_tuple = std::get<0>(result);
      auto y_tuple = std::get<1>(result);
      auto out = std::get<2>(result);
      auto ref_out = std::get<3>(result);

      // Extract elements from the inner tuples
      auto x_fp8 = std::get<0>(x_tuple);
      auto x_scales_aligned = std::get<1>(x_tuple);
      auto y_fp8 = std::get<0>(y_tuple);
      auto y_scales = std::get<1>(y_tuple);

      std::cout << "Test data constructed successfully." << std::endl;
      std::cout << "x_fp8 shape: [" << x_fp8.sizes() << "]" << std::endl;
      std::cout << "y_fp8 shape: [" << y_fp8.sizes() << "]" << std::endl;
      std::cout << "out shape: [" << out.sizes() << "]" << std::endl;
      std::cout << "ref_out shape: [" << ref_out.sizes() << "]" << std::endl;

      deepgemm_aot_wrapper.Forward(x_fp8.data_ptr(), x_scales_aligned.data_ptr(), y_fp8.data_ptr(), y_scales.data_ptr(),
                                   out.data_ptr(), compute_m, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      // Calculate cosine similarity between out and ref_out
      // Flatten tensors into vectors
      auto out_flat = out.view(-1);
      auto ref_out_flat = ref_out.view(-1);

      // Calculate dot product
      auto dot_product = torch::sum(out_flat * ref_out_flat);

      // Calculate L2 norms
      auto out_norm = torch::sqrt(torch::sum(out_flat * out_flat));
      auto ref_out_norm = torch::sqrt(torch::sum(ref_out_flat * ref_out_flat));

      // Calculate cosine similarity
      auto cosine_similarity = dot_product / (out_norm * ref_out_norm);

      // Calculate cosine similarity difference (compared to perfect match)
      auto cosine_diff = 1.0 - cosine_similarity;

      std::cout << compute_m << "Cosine similarity between out and ref_out: " << cosine_similarity.item<float>()
                << std::endl;
      std::cout << compute_m << "Cosine similarity difference: " << cosine_diff.item<float>() << std::endl;

      EXPECT_TRUE(cosine_diff.item<float>() < 0.01)
          << "Cosine similarity difference is too large: " << cosine_diff.item<float>();
    }
  }
};

/**
 * @brief Test case for common GEMM operations using DeepGEMM AOT wrapper
 *
 * This test verifies the functionality of the DeepGEMM AOT wrapper
 * for matrix multiplication operations. It requires a CUDA-capable GPU.
 */
TEST_F(DeepGEMMAOTWrapperTestSuit, CommonGemmTest) {
  // Initialize CUDA device
  torch::Device device(torch::kCUDA);
  if (torch::cuda::is_available()) {
    std::cerr << "CUDA is available! Testing on GPU." << std::endl;
  } else {
    GTEST_SKIP() << "CUDA is not available. This test requires GPU." << std::endl;
  }

  // Run the test function
  TestDeepGEMMAOTWrapper();
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels