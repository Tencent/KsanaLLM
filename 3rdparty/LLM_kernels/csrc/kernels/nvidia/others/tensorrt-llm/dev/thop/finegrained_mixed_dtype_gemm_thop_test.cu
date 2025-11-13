/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * Adapted from [TensorRT-LLM Project]
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc4/tests/unittest/_torch/thop/parallel/test_finegrained_mixed_dtype_gemm.py
 *
 */

#include <gtest/gtest.h>

#include <iostream>
#include <limits>
#include <sstream>

#include <fmt/format.h>
#include <torch/torch.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/cudaFp8Utils.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/finegrained_mixed_dtype_gemm_thop.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/torch_utils.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"

using namespace llm_kernels::nvidia;
using namespace llm_kernels::nvidia::tensorrt_llm::dev;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaFinegrainedMixedDtypeGemmTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    // 固定随机数种子
    torch::manual_seed(42);
    torch::cuda::manual_seed_all(42);
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const size_t warmup = 50;
  const size_t iters = 200;

 protected:
  int64_t get_tflops(int64_t M, int64_t N, int64_t K, double ms) { return 2 * M * N * K / ms * 1e-9; }

  torch::Tensor unpack_int4_packed_tensor_to_int8(torch::Tensor weight) {
    KLLM_KERNEL_CHECK_WITH_INFO(weight.numel() != 0, "weight should not be empty tensor");
    KLLM_KERNEL_CHECK_WITH_INFO(weight.dtype() == torch::kInt8, "Weight must be a packed int8 tensor");

    std::vector<int64_t> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
      int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    torch::Tensor unpacked_weight =
        torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr = static_cast<int8_t*>(weight.data_ptr());
    int8_t* unpacked_ptr = static_cast<int8_t*>(unpacked_weight.data_ptr());

    for (int64_t packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
      int8_t packed_data = packed_ptr[packed_idx];

      int8_t elt_0 = (int8_t(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
      int8_t elt_1 = packed_data >> 4;

      unpacked_ptr[2 * packed_idx + 0] = elt_0;
      unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
  }

  torch::Tensor preprocess_weights_for_mixed_gemm(torch::Tensor tensor, torch::ScalarType quant_mode,
                                                  torch::ScalarType act_dtype, int sm_ = -1,
                                                  bool do_weight_interleave = true) {
    if (sm_ <= 0) {
      sm_ = GetSMVersion();
    }

    bool was_2d = false;
    if (tensor.dim() == 2) {
      tensor = tensor.unsqueeze(0);
      was_2d = true;
    } else if (sm_ >= 90) {
      sm_ = 80;
    }
    if (sm_ > 90) {
      sm_ = 80;
    }

    std::map<std::string, std::vector<int>> permutation_map = {
        {"16_8", {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}},
        {"16_4", {0, 1, 8,  9,  16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
                  4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31}},
        {"8_4", {0, 1, 2,  3,  16, 17, 18, 19, 4,  5,  6,  7,  20, 21, 22, 23,
                 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31}}};

    int BITS_PER_ELT_A = (act_dtype == torch::kFloat8_e4m3fn) ? 8 : 16;
    int BITS_PER_ELT_B = (quant_mode == torch::kQUInt4x2) ? 4 : 8;
    int MMA_SHAPE_N = 8;
    int B_ROWS_PER_MMA = 8 * 16 / BITS_PER_ELT_B;

    int64_t num_experts = tensor.size(0);
    int64_t num_rows = tensor.size(1);
    int64_t num_cols = tensor.size(2);

    KLLM_KERNEL_CHECK_WITH_INFO(sm_ >= 75, "SM version must be >= 75");
    KLLM_KERNEL_CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0, "num_rows must be divisible by B_ROWS_PER_MMA");
    KLLM_KERNEL_CHECK_WITH_INFO(num_cols % MMA_SHAPE_N == 0, "num_cols must be divisible by MMA_SHAPE_N");

    auto original_shape = tensor.sizes().vec();

    // 第一步：permute_B_rows_for_mixed_gemm
    if (do_weight_interleave) {
      std::string key = std::to_string(BITS_PER_ELT_A) + "_" + std::to_string(BITS_PER_ELT_B);
      auto& perm = permutation_map[key];

      std::vector<int64_t> row_idx_list;
      for (int64_t row_idx = 0; row_idx < num_rows; row_idx++) {
        int64_t new_idx = (row_idx / B_ROWS_PER_MMA) * B_ROWS_PER_MMA + perm[row_idx % B_ROWS_PER_MMA];
        row_idx_list.push_back(new_idx);
      }

      tensor =
          tensor.index({torch::indexing::Slice(), torch::tensor(row_idx_list, torch::kLong), torch::indexing::Slice()});
    }

    // 第二步：subbyte_transpose
    if (BITS_PER_ELT_B == 4) {
      tensor = tensor.view(torch::kUInt8);

      auto high_tensor = torch::bitwise_right_shift(tensor, 4).permute({0, 2, 1}).unsqueeze(2);
      auto low_tensor =
          torch::bitwise_right_shift(torch::bitwise_left_shift(tensor, 4), 4).permute({0, 2, 1}).unsqueeze(2);

      auto new_tensor = torch::cat({low_tensor, high_tensor}, 2).reshape({tensor.size(0), -1, tensor.size(1)});

      new_tensor = new_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(0, torch::indexing::None, 2)}) +
                   new_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                     torch::indexing::Slice(1, torch::indexing::None, 2)}) *
                       16;

      tensor = new_tensor.view(torch::kInt8).reshape(original_shape);
    } else {
      tensor = tensor.permute({0, 2, 1}).reshape(original_shape);
    }

    // 第三步：interleave_column_major_tensor
    if (do_weight_interleave) {
      int interleave = BITS_PER_ELT_A / BITS_PER_ELT_B;

      if (interleave > 1 && sm_ < 90) {
        int rows_per_tile = 128 * 8 / BITS_PER_ELT_A;
        int elts_in_int32 = 32 / BITS_PER_ELT_B;

        KLLM_KERNEL_CHECK_WITH_INFO(num_rows % elts_in_int32 == 0, "num_rows must be divisible by elts_in_int32");
        KLLM_KERNEL_CHECK_WITH_INFO(num_rows % rows_per_tile == 0, "num_rows must be divisible by rows_per_tile");

        tensor =
            tensor.reshape({num_experts, -1, interleave, num_rows / rows_per_tile, rows_per_tile * 4 / elts_in_int32});
        tensor = tensor.permute({0, 1, 3, 2, 4}).reshape(original_shape);
      }

      // 第四步：add_bias_and_interleave_quantized_tensor_inplace
      if (BITS_PER_ELT_B == 8) {
        auto mask = (tensor > 127).to(torch::kUInt8);
        tensor = tensor + (-256 * mask) + 128;

        tensor = tensor.reshape({-1, 4})
                     .index({torch::indexing::Slice(), torch::tensor({0, 2, 1, 3})})
                     .reshape(tensor.sizes());

      } else if (BITS_PER_ELT_B == 4) {
        tensor = tensor.view(torch::kUInt8);

        auto high_tensor = torch::bitwise_right_shift(tensor, 4).unsqueeze(-1);
        auto low_tensor = torch::bitwise_right_shift(torch::bitwise_left_shift(tensor, 4), 4).unsqueeze(-1);

        auto new_tensor = torch::cat({low_tensor, high_tensor}, -1).reshape({tensor.size(0), tensor.size(1), -1});

        new_tensor = new_tensor.reshape({-1, 8})
                         .index({torch::indexing::Slice(), torch::tensor({0, 2, 4, 6, 1, 3, 5, 7})})
                         .reshape(new_tensor.sizes());

        auto mask = (new_tensor > 7).to(torch::kUInt8);
        new_tensor = new_tensor + (-16 * mask) + 8;

        new_tensor = new_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(0, torch::indexing::None, 2)}) +
                     new_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(1, torch::indexing::None, 2)}) *
                         16;

        tensor = new_tensor.view(torch::kInt8);
      } else {
        throw std::runtime_error("Unsupported BITS_PER_ELT_B");
      }
    }

    if (was_2d) {
      return tensor.squeeze(0).contiguous();
    }
    return tensor.contiguous();
  }

  template <typename dtype>
  torch::Tensor finegrained_mixed_dtype_gemm(int64_t M, int64_t N, int64_t K, torch::Tensor& input,
                                             torch::Tensor& weight, torch::Tensor& scales, int64_t group_size,
                                             double alpha, std::optional<torch::Tensor> zeros = std::nullopt) {
    torch::Tensor output =
        torch::randn({M, N}, torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA));

    // 构建算子，quant_mode 根据是否有 zeros 来决定
    int quant_mode = zeros.has_value() ? 1 : 0;
    auto runner = FinegrainedMixedDtypeGemmRunner(ScalarType::Float8_e4m3fn, GetScalarType<dtype>(), quant_mode);

    // 开辟workspace
    std::vector<size_t> A_shape(input.sizes().begin(), input.sizes().end());
    std::vector<size_t> B_shape(weight.sizes().begin(), weight.sizes().end());
    size_t workspace_size = runner.getWorkspaceSize(A_shape, B_shape);
    torch::Tensor workspace = torch::zeros({static_cast<int64_t>(workspace_size)},
                                           torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));

    // 构建输入输出
    Tensor workspace_tensor(workspace.data_ptr(),
                            std::vector<size_t>(workspace.sizes().begin(), workspace.sizes().end()), ScalarType::Int8);
    Tensor input_tensor(input.data_ptr(), std::vector<size_t>(input.sizes().begin(), input.sizes().end()),
                        GetScalarType<__nv_fp8_e4m3>());
    Tensor weight_tensor(weight.data_ptr(), std::vector<size_t>(weight.sizes().begin(), weight.sizes().end()),
                         GetScalarType<int8_t>());
    Tensor scales_tensor(scales.data_ptr(), std::vector<size_t>(scales.sizes().begin(), scales.sizes().end()),
                         GetScalarType<half>());
    Tensor output_tensor(output.data_ptr(), std::vector<size_t>(output.sizes().begin(), output.sizes().end()),
                         GetScalarType<dtype>());

    // 处理 zeros tensor - zeros的dtype应该是half(FP16)，不是dtype
    std::optional<Tensor> zeros_tensor = std::nullopt;
    if (zeros.has_value()) {
      zeros_tensor =
          Tensor(zeros.value().data_ptr(),
                 std::vector<size_t>(zeros.value().sizes().begin(), zeros.value().sizes().end()), ScalarType::Float16);
    }

    // gemm计算
    runner.runGemm(stream, output_tensor, workspace_tensor, input_tensor, weight_tensor, scales_tensor, group_size, -1,
                   std::nullopt, zeros_tensor, alpha);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    return output;
  }

  bool woq_assert_near_eq(const torch::Tensor& ref, const torch::Tensor& act, int wTypeId) {
    // Match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    int bits_in_type;
    if (wTypeId == 1) {
      bits_in_type = 8;
    } else {
      bits_in_type = 4;
    }

    float quant_range_scale = 1.0f / static_cast<float>(1 << (bits_in_type - 1));

    // Get the maximum absolute value
    float max_val = torch::max(torch::abs(ref)).item<float>();

    // Calculate absolute tolerance (allow for rounding)
    float atol = (max_val * quant_range_scale) * 1.5f;
    float rtol = 1e-7f;

    // Check if tensors are close
    return torch::allclose(ref, act, rtol, atol);
  }

  template <typename dtype>
  void TestFinegrainedMixedDtypeGemmPrecision(const int64_t M, const int64_t N, const int64_t K,
                                              const bool has_pre_quant, const bool has_zero) {
    // 测试参数：has_bias=false, use_w4a8_awq=true
    const int64_t group_size = 128;
    int64_t total_groups = (K + group_size - 1) / group_size;

    // 对于W4A8 AWQ，scale和zero的dtype是float16
    torch::ScalarType scale_zero_dtype = torch::kFloat16;
    torch::ScalarType activation_dtype = GetTorchDataType<dtype>();
    torch::ScalarType activation_type = torch::kFloat8_e4m3fn;

    // 构建数据
    torch::Tensor activation =
        torch::randn({M, K}, torch::TensorOptions().dtype(activation_dtype).device(torch::kCUDA));
    torch::Tensor pre_quant_scale =
        torch::rand({1, K}, torch::TensorOptions().dtype(activation_dtype).device(torch::kCUDA));
    torch::Tensor scale =
        torch::rand({total_groups, N}, torch::TensorOptions().dtype(scale_zero_dtype).device(torch::kCUDA));
    torch::Tensor zero =
        torch::randn({total_groups, N}, torch::TensorOptions().dtype(scale_zero_dtype).device(torch::kCUDA));
    torch::Tensor fp8_alpha = torch::rand({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    int64_t num_weights_in_32_bits = 8;  // for torch.quint4x2
    torch::Tensor unprocessed_int_weight = torch::randint(
        static_cast<int64_t>(-std::pow(2, 31)), static_cast<int64_t>(std::pow(2, 31)), {K, N / num_weights_in_32_bits},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor unprocessed_weight = unprocessed_int_weight.view(torch::kInt8);

    // pre_quant_scale
    if (has_pre_quant) {
      torch::Tensor pre_quant_scale_expanded = pre_quant_scale.repeat({M, 1});
      activation = torch::mul(activation, pre_quant_scale_expanded);
    }

    // kernel计算
    // -- 预处理权重
    torch::Tensor cuda_q_weight =
        preprocess_weights_for_mixed_gemm(unprocessed_weight.cpu(), torch::kQUInt4x2, activation_type)
            .cuda()
            .contiguous();
    // -- 将activation转换为FP8用于实际计算
    torch::Tensor fp8_activation = activation.to(activation_type).contiguous();
    // -- 调用GEMM内核
    std::optional<torch::Tensor> zero_opt = has_zero ? std::optional<torch::Tensor>(zero) : std::nullopt;
    torch::Tensor output = finegrained_mixed_dtype_gemm<dtype>(M, N, K, fp8_activation, cuda_q_weight, scale,
                                                               group_size, fp8_alpha.item<float>(), zero_opt);

    // 计算参考结果
    // -- scale解包
    torch::Tensor scale_ref =
        scale.repeat_interleave(group_size, 0).index({torch::indexing::Slice(0, K), torch::indexing::Slice()});
    // -- 权重解包
    torch::Tensor ref_q_weight = unpack_int4_packed_tensor_to_int8(unprocessed_weight.cpu()).contiguous().cuda();
    // -- 反量化权重
    torch::Tensor ref_th_weight = ref_q_weight.to(activation_dtype) * scale_ref.to(activation_dtype);
    // -- 零点处理
    if (has_zero) {
      torch::Tensor zero_ref =
          zero.repeat_interleave(group_size, 0).index({torch::indexing::Slice(0, K), torch::indexing::Slice()});
      ref_th_weight += zero_ref.to(activation_dtype);
    }
    // -- 注意：在W4A8 AWQ中，activation需要乘以alpha，但需要保持相同的dtype
    torch::Tensor activation_scaled = (activation * fp8_alpha).to(activation_dtype);
    torch::Tensor ref_output = torch::matmul(activation_scaled, ref_th_weight);

    // 结果检查
    bool all_close = woq_assert_near_eq(ref_output.to(torch::kFloat32), output.to(torch::kFloat32), 2);
    EXPECT_TRUE(all_close);
  }

  void TestFinegrainedMixedDtypeGemmPerformance(const int64_t M, const int64_t N, const int64_t K) {
    const int64_t GROUP_SIZE = 128;

    // 创建数据
    torch::Tensor activation =
        torch::randn({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    torch::Tensor weight = torch::randint(0, static_cast<int64_t>(std::pow(2, 8) - 1), {K, N / 2},
                                          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    torch::Tensor pre_quant_scale =
        torch::rand({K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    torch::Tensor weight_scale =
        torch::rand({K / GROUP_SIZE, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    torch::Tensor weight_scale_2 = torch::rand({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor input_scale =
        torch::tensor({1.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // 权重处理
    torch::Tensor processed_w = preprocess_weights_for_mixed_gemm(weight.to(torch::kInt8).contiguous().cpu(),
                                                                  torch::kQUInt4x2, torch::kFloat8_e4m3fn)
                                    .cuda()
                                    .contiguous();
    torch::Tensor processed_weight_scale = (weight_scale / weight_scale_2).to(torch::kFloat16).contiguous();
    double alpha = (input_scale.to(torch::kFloat) * weight_scale_2.to(torch::kFloat)).item<double>();

    // 构建算子
    auto runner = FinegrainedMixedDtypeGemmRunner(ScalarType::Float8_e4m3fn, ScalarType::BFloat16, 0);
    std::vector<size_t> A_shape(activation.sizes().begin(), activation.sizes().end());
    std::vector<size_t> B_shape(processed_w.sizes().begin(), processed_w.sizes().end());
    size_t workspace_size = runner.getWorkspaceSize(A_shape, B_shape);
    BufferMeta workspace = CreateBuffer<char>(MemoryType::MEMORY_GPU, {workspace_size}, false);

    // 构建输入输出
    torch::Tensor output = torch::randn({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    torch::Tensor quantized_x =
        torch::empty(activation.sizes(), torch::dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
    Tensor quantized_x_tensor(quantized_x.data_ptr(),
                              std::vector<size_t>(quantized_x.sizes().begin(), quantized_x.sizes().end()),
                              GetScalarType<__nv_fp8_e4m3>());
    Tensor workspace_tensor(workspace.data_ptr, workspace.shape, GetScalarType<char>());
    Tensor w_tensor(processed_w.data_ptr(), std::vector<size_t>(processed_w.sizes().begin(), processed_w.sizes().end()),
                    GetScalarType<int8_t>());
    Tensor weight_scale_tensor(
        processed_weight_scale.data_ptr(),
        std::vector<size_t>(processed_weight_scale.sizes().begin(), processed_weight_scale.sizes().end()),
        GetScalarType<half>());
    Tensor output_tensor(output.data_ptr(), std::vector<size_t>(output.sizes().begin(), output.sizes().end()),
                         ScalarType::BFloat16);

    // 默认gemm计算
    torch::Tensor x = activation.to(torch::kBFloat16);

    llm_kernels::nvidia::tensorrt_llm::dev::common::invokeQuantizeMatrix(
        static_cast<__nv_fp8_e4m3*>(quantized_x.data_ptr()), static_cast<float*>(input_scale.data_ptr()),
        static_cast<const __nv_bfloat16*>(x.data_ptr()), x.numel(), x.size(-1),
        llm_kernels::nvidia::tensorrt_llm::dev::common::QuantizeMode::PER_TENSOR, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto default_cuda_run = [&]() {
      runner.runGemm(stream, output_tensor, workspace_tensor, quantized_x_tensor, w_tensor, weight_scale_tensor,
                     GROUP_SIZE, -1, std::nullopt, std::nullopt, alpha);
    };
    float default_time_elapsed_ms = MeasureCudaExecutionTime(default_cuda_run, stream, warmup, iters);
    int64_t default_tflops = get_tflops(M, N, K, default_time_elapsed_ms);

    // config gemm计算
    int64_t num_configs = runner.getNumConfigs();
    std::vector<float> config_times(num_configs, std::numeric_limits<float>::max());
    for (int64_t config_idx = 0; config_idx < num_configs; config_idx++) {
      auto cuda_run = [&]() {
        runner.runGemm(stream, output_tensor, workspace_tensor, quantized_x_tensor, w_tensor, weight_scale_tensor,
                       GROUP_SIZE, config_idx, std::nullopt, std::nullopt, alpha);
      };
      config_times[config_idx] = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);
    }
    float best_time_elapsed_ms = *std::min_element(config_times.begin(), config_times.end());
    int64_t best_tflops = get_tflops(M, N, K, best_time_elapsed_ms);

    std::cout << fmt::format("MNK({},{},{}) default {} ms, {} TFLOPS, best {} ms, {} TFLOPS\n", M, N, K,
                             default_time_elapsed_ms, default_tflops, best_time_elapsed_ms, best_tflops);
  }
};

TEST_F(NvidiaFinegrainedMixedDtypeGemmTestSuit, TestFinegrainedMixedDtypeGemmPrecision) {
  // pre_quant + zero
  TestFinegrainedMixedDtypeGemmPrecision<half>(512, 2048, 1024, true, true);
  TestFinegrainedMixedDtypeGemmPrecision<__nv_bfloat16>(512, 2048, 1024, true, true);
  // pre_quant + no zero
  TestFinegrainedMixedDtypeGemmPrecision<half>(512, 2048, 1024, true, false);
  TestFinegrainedMixedDtypeGemmPrecision<__nv_bfloat16>(512, 2048, 1024, true, false);
  // no pre_quant + no zero
  TestFinegrainedMixedDtypeGemmPrecision<half>(512, 2048, 1024, false, false);
  TestFinegrainedMixedDtypeGemmPrecision<__nv_bfloat16>(512, 2048, 1024, false, false);
}

TEST_F(NvidiaFinegrainedMixedDtypeGemmTestSuit, TestFinegrainedMixedDtypeGemmPerformance) {
  std::vector<std::tuple<int64_t, int64_t, int64_t>> testcases = {
      {1, 25600, 5120},   {2, 25600, 5120},   {4, 25600, 5120},    {8, 25600, 5120},
      {16, 25600, 5120},  {32, 25600, 5120},  {64, 25600, 5120},   {128, 25600, 5120},
      {256, 25600, 5120}, {512, 25600, 5120}, {1024, 25600, 5120}, {2048, 25600, 5120},
  };
  for (auto& tc : testcases) {
    TestFinegrainedMixedDtypeGemmPerformance(std::get<0>(tc), std::get<1>(tc), std::get<2>(tc));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels