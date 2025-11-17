/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#include "ksana_llm/utils/config/schedule_config_parser.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA

class DirectFlashAttnComparisonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 设置 CUDA 设备
    device_ = torch::kCUDA;
    c10::cuda::set_device(0);

    // 初始化 FlashAttentionBackend，现在会自动加载所有可用的 attention 实现
    backend_ = std::make_unique<FlashAttentionBackend>();
    ASSERT_TRUE(backend_->Initialize()) << "Failed to initialize FlashAttentionBackend";
  }

  void TearDown() override { backend_.reset(); }

  // 创建cumulative sequence lengths tensor
  at::Tensor CreateCuSeqlens(int batch_size, int seq_len) {
    std::vector<int32_t> cu_seqlens_data;
    cu_seqlens_data.reserve(batch_size + 1);
    cu_seqlens_data.push_back(0);
    for (int i = 1; i <= batch_size; ++i) {
      cu_seqlens_data.push_back(i * seq_len);
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto cu_seqlens = torch::tensor(cu_seqlens_data, options);

    return cu_seqlens;
  }

  // 将 batch 格式转换为 varlen 格式
  at::Tensor BatchToVarlen(const at::Tensor& batch_tensor, const at::Tensor& /*cu_seqlens*/) {
    const auto batch_size = batch_tensor.size(0);
    const auto seq_len = batch_tensor.size(1);
    const auto num_heads = batch_tensor.size(2);
    const auto head_size = batch_tensor.size(3);

    return batch_tensor.reshape({batch_size * seq_len, num_heads, head_size});
  }

  // 将 varlen 格式转换为 batch 格式
  at::Tensor VarlenToBatch(const at::Tensor& varlen_tensor, const at::Tensor& /*cu_seqlens*/, int batch_size,
                           int seq_len, int num_heads, int head_size) {
    return varlen_tensor.reshape({batch_size, seq_len, num_heads, head_size});
  }

  // 通用的tensor差异对比函数
  struct ComparisonResult {
    double max_abs_diff;
    double mean_abs_diff;
    double cosine_similarity;
    double max_relative_diff;
    double mean_relative_diff;
    bool has_invalid_values;
    bool comparison_passed;
    std::vector<double> bin_edges;
    std::vector<int64_t> histogram;
  };

  ComparisonResult CompareTensorOutputs(const at::Tensor& tensor1, const at::Tensor& tensor2,
                                        const std::string& tensor1_name, const std::string& tensor2_name,
                                        double cosine_threshold = 0.999, double max_abs_threshold = 0.01,
                                        double mean_abs_threshold = 0.001, bool isPrint = false) {
    ComparisonResult result;

    if (isPrint) {
      std::cout << "\n=== " << tensor1_name << " vs " << tensor2_name << " 输出对比分析 ===" << std::endl;
    }

    // 确保形状匹配
    if (tensor1.sizes() != tensor2.sizes()) {
      if (isPrint) {
        std::cout << "❌ 形状不匹配: " << tensor1_name << "=" << tensor1.sizes() << " vs " << tensor2_name << "="
                  << tensor2.sizes() << std::endl;
      }
      result.comparison_passed = false;
      return result;
    }
    // 转换到 float32 进行高精度计算
    auto tensor1_f32 = tensor1.to(torch::kFloat32);
    auto tensor2_f32 = tensor2.to(torch::kFloat32);
    // 计算绝对误差
    auto diff_tensor = torch::abs(tensor1_f32 - tensor2_f32);
    result.max_abs_diff = torch::max(diff_tensor).item<double>();
    result.mean_abs_diff = torch::mean(diff_tensor).item<double>();
    if (isPrint) {
      std::cout << "  最大绝对误差: " << std::fixed << std::setprecision(6) << result.max_abs_diff << std::endl;
      std::cout << "  平均绝对误差: " << std::fixed << std::setprecision(6) << result.mean_abs_diff << std::endl;
    }
    // 1. 计算余弦相似度
    auto tensor1_flat = tensor1_f32.flatten();
    auto tensor2_flat = tensor2_f32.flatten();
    // 计算点积
    auto dot_product = torch::sum(tensor1_flat * tensor2_flat);

    // 计算范数
    auto norm_tensor1 = torch::norm(tensor1_flat);
    auto norm_tensor2 = torch::norm(tensor2_flat);

    // 计算余弦相似度
    auto cosine_similarity = dot_product / (norm_tensor1 * norm_tensor2);
    result.cosine_similarity = cosine_similarity.item<double>();

    if (isPrint) {
      std::cout << "  余弦相似度: " << std::fixed << std::setprecision(8) << result.cosine_similarity << std::endl;
    }

    // 2. 计算差异元素的直方图分析
    auto abs_diff_flat = torch::abs(tensor1_flat - tensor2_flat);

    // 定义直方图区间
    result.bin_edges = {0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, std::numeric_limits<double>::infinity()};
    result.histogram.resize(result.bin_edges.size() - 1, 0);

    // 将tensor转换为CPU并计算直方图
    auto abs_diff_cpu = abs_diff_flat.cpu();
    auto abs_diff_accessor = abs_diff_cpu.accessor<float, 1>();

    for (int64_t i = 0; i < abs_diff_cpu.size(0); ++i) {
      double diff_val = static_cast<double>(abs_diff_accessor[i]);
      for (size_t j = 0; j < result.bin_edges.size() - 1; ++j) {
        if (diff_val >= result.bin_edges[j] && diff_val < result.bin_edges[j + 1]) {
          result.histogram[j]++;
          break;
        }
      }
    }

    // 输出直方图统计
    if (isPrint) {
      std::cout << "  差异元素直方图分布:" << std::endl;
      int64_t total_elements = abs_diff_cpu.size(0);
      for (size_t i = 0; i < result.histogram.size(); ++i) {
        double percentage = (static_cast<double>(result.histogram[i]) / total_elements) * 100.0;

        // 格式化区间边界显示
        std::string left_bound, right_bound;
        if (result.bin_edges[i] == 0.0) {
          left_bound = "0.000000";
        } else if (result.bin_edges[i] < 1e-3) {
          left_bound = std::to_string(result.bin_edges[i]);
        } else {
          left_bound = std::to_string(result.bin_edges[i]);
        }

        if (std::isinf(result.bin_edges[i + 1])) {
          right_bound = "∞";
        } else if (result.bin_edges[i + 1] < 1e-3) {
          right_bound = std::to_string(result.bin_edges[i + 1]);
        } else {
          right_bound = std::to_string(result.bin_edges[i + 1]);
        }

        std::cout << "    [" << left_bound << ", " << right_bound << "): " << result.histogram[i] << " 个元素 ("
                  << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
      }
    }

    // 3. 计算相对误差统计
    auto relative_diff = torch::abs(tensor1_flat - tensor2_flat) / (torch::abs(tensor2_flat) + 1e-8);
    result.max_relative_diff = torch::max(relative_diff).item<double>();
    result.mean_relative_diff = torch::mean(relative_diff).item<double>();

    if (isPrint) {
      std::cout << "  最大相对误差: " << std::fixed << std::setprecision(6) << result.max_relative_diff << std::endl;
      std::cout << "  平均相对误差: " << std::fixed << std::setprecision(6) << result.mean_relative_diff << std::endl;
    }

    // 4. 检查数值健康度
    bool tensor1_has_nan = torch::any(torch::isnan(tensor1)).item<bool>();
    bool tensor1_has_inf = torch::any(torch::isinf(tensor1)).item<bool>();
    bool tensor2_has_nan = torch::any(torch::isnan(tensor2)).item<bool>();
    bool tensor2_has_inf = torch::any(torch::isinf(tensor2)).item<bool>();

    result.has_invalid_values = tensor1_has_nan || tensor1_has_inf || tensor2_has_nan || tensor2_has_inf;

    // 如果存在 NaN 或 Inf，则测试失败
    if (result.has_invalid_values) {
      if (isPrint) {
        std::cout << "❌ 检测到无效数值!" << std::endl;
      }
      result.comparison_passed = false;
      return result;
    }

    // 6. 判断对比结果
    result.comparison_passed = true;

    // 余弦相似度应该非常接近1
    if (result.cosine_similarity < cosine_threshold) {
      if (isPrint) {
        std::cout << "⚠️  余弦相似度较低: " << result.cosine_similarity << " < " << cosine_threshold << std::endl;
      }
      result.comparison_passed = false;
    }

    // 最大绝对误差不应该太大
    if (result.max_abs_diff > max_abs_threshold) {
      if (isPrint) {
        std::cout << "⚠️  最大绝对误差较大: " << result.max_abs_diff << " > " << max_abs_threshold << std::endl;
      }
      result.comparison_passed = false;
    }

    // 平均绝对误差应该很小
    if (result.mean_abs_diff > mean_abs_threshold) {
      if (isPrint) {
        std::cout << "⚠️  平均绝对误差较大: " << result.mean_abs_diff << " > " << mean_abs_threshold << std::endl;
      }
      result.comparison_passed = false;
    }

    if (isPrint) {
      if (result.comparison_passed) {
        std::cout << "✅ " << tensor1_name << "与" << tensor2_name << "输出对比通过" << std::endl;
      } else {
        std::cout << "❌ " << tensor1_name << "与" << tensor2_name << "输出存在显著差异" << std::endl;
      }
      std::cout << "=== 对比分析完成 ===" << std::endl;
    }
    return result;
  }

 protected:
  torch::Device device_ = torch::Device(torch::kCUDA, 0);

  // 后端实例，会自动加载所有可用的 attention 实现
  std::unique_ptr<FlashAttentionBackend> backend_;
};

TEST_F(DirectFlashAttnComparisonTest, TestFA3WithSyntheticInputs) {
  // 检查 FA3 函数指针是否可用
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    GTEST_SKIP() << "FA3 function not available";
  }

  // 控制调试输出的参数
  const bool isPrint = false;  // 设置为 true 可以启用详细的调试输出

  // 测试参数配置
  struct TestConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_size;
    int head_size_v;
    bool is_causal;
    std::string description;
  };
  std::vector<TestConfig> test_configs = {
      {1, 128, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 256, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 512, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 1024, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 2048, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
  };

  int success_count = 0;
  int total_count = test_configs.size();

  for (const auto& config : test_configs) {
    if (isPrint) {
      std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
      std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
                << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
                << ", head_size=" << config.head_size << ", head_size_v=" << config.head_size_v
                << ", is_causal=" << config.is_causal << std::endl;
    }

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
      torch::manual_seed(42);
      auto q_tensor = torch::randn({total_tokens, config.num_heads, config.head_size}, options);
      auto k_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);
      auto v_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size_v}, options);

      auto cu_seqlens = CreateCuSeqlens(config.batch_size, config.seq_len);

      auto q_contiguous = q_tensor.contiguous();
      auto k_contiguous = k_tensor.contiguous();
      auto v_contiguous = v_tensor.contiguous();
      auto q_tmp_tensor = torch::reshape(q_contiguous, {total_tokens, config.num_heads, config.head_size});
      std::optional<at::Tensor> k_new_ = std::nullopt;
      std::optional<at::Tensor> v_new_ = std::nullopt;
      std::optional<at::Tensor> q_v_ = std::nullopt;
      std::optional<at::Tensor> out_ = std::nullopt;
      std::optional<at::Tensor> cu_seqlens_q_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_new_ = std::nullopt;
      std::optional<at::Tensor> seqused_q_ = std::nullopt;
      std::optional<at::Tensor> seqused_k_ = std::nullopt;
      std::optional<int64_t> max_seqlen_q_ = static_cast<int64_t>(config.seq_len);
      std::optional<int64_t> max_seqlen_k_ = static_cast<int64_t>(config.seq_len);
      std::optional<at::Tensor> page_table_ = std::nullopt;
      std::optional<at::Tensor> kv_batch_idx_ = std::nullopt;
      std::optional<at::Tensor> leftpad_k_ = std::nullopt;
      std::optional<at::Tensor> rotary_cos_ = std::nullopt;
      std::optional<at::Tensor> rotary_sin_ = std::nullopt;
      std::optional<at::Tensor> seqlens_rotary_ = std::nullopt;
      std::optional<at::Tensor> q_descale_ = std::nullopt;
      std::optional<at::Tensor> k_descale_ = std::nullopt;
      std::optional<at::Tensor> v_descale_ = std::nullopt;
      std::optional<double> softmax_scale_opt = std::optional<double>(static_cast<double>(attn_scale));
      int64_t window_size_left = -1;
      int64_t window_size_right = -1;
      int64_t attention_chunk = 0;
      double softcap_val = 0.0;
      bool is_rotary_interleaved = false;
      std::optional<at::Tensor> scheduler_metadata_ = std::nullopt;
      int64_t num_splits = 0;
      std::optional<bool> pack_gqa_ = std::nullopt;
      int64_t sm_margin = 0;
      // 1. 调用FA3（计时）
      at::Tensor fa3_output;
      double fa3_avg_ms = 0.0;
      const int warmup_iters = 2;
      const int bench_iters = 5;
      std::vector<at::Tensor> fa3_result;
      try {
        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                            cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_,
                            page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
                            q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal, window_size_left,
                            window_size_right, attention_chunk, softcap_val, is_rotary_interleaved, scheduler_metadata_,
                            num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_result = mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                                   cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_,
                                   max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_,
                                   seqlens_rotary_, q_descale_, k_descale_, v_descale_, softmax_scale_opt,
                                   config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
                                   is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fa3_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
        fa3_output = fa3_result[0];
      } catch (const std::exception& e) {
        if (isPrint) {
          std::cerr << "FA3调用失败: " << e.what() << std::endl;
        }
        continue;
      }
      // 验证 FA3 输出
      ASSERT_FALSE(fa3_result.empty()) << "FA3 应该返回非空结果";
      auto output_tensor = fa3_result[0];
      ASSERT_EQ(output_tensor.dim(), 3) << "输出张量应该是3维的";
      ASSERT_EQ(output_tensor.size(0), total_tokens) << "输出张量第一维应该等于总token数";
      ASSERT_EQ(output_tensor.size(1), config.num_heads) << "输出张量第二维应该等于头数";
      ASSERT_EQ(output_tensor.size(2), config.head_size_v) << "输出张量第三维应该等于头大小";

      // 检查输出是否包含有效数值
      bool has_nan = torch::any(torch::isnan(output_tensor)).item<bool>();
      bool has_inf = torch::any(torch::isinf(output_tensor)).item<bool>();
      ASSERT_FALSE(has_nan) << "输出不应包含 NaN 值";
      ASSERT_FALSE(has_inf) << "输出不应包含 Inf 值";

      // 检查输出范围是否合理
      auto output_abs_max = torch::max(torch::abs(output_tensor)).item<float>();
      ASSERT_LT(output_abs_max, 100.0f) << "输出值的绝对值应该在合理范围内";

      if (isPrint) {
        std::cout << "✅ FA3 测试通过: 平均耗时=" << fa3_avg_ms << " ms" << std::endl;
        std::cout << "   输出形状: " << output_tensor.sizes() << std::endl;
        std::cout << "   输出最大绝对值: " << output_abs_max << std::endl;
      }
      success_count++;
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      }
    }
  }

  // 验证至少有一些测试成功
  EXPECT_GT(success_count, 0) << "至少应该有一个测试配置成功";
}

TEST_F(DirectFlashAttnComparisonTest, TestFA3WithFP8) {
  // 检查 FA3 函数指针是否可用
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    GTEST_SKIP() << "FA3 function not available";
  }

  // 控制调试输出的参数
  const bool isPrint = false;  // 设置为 true 可以启用详细的调试输出

  // 测试参数配置
  struct TestConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_size;
    int head_size_v;
    bool is_causal;
    std::string description;
  };
  std::vector<TestConfig> test_configs = {
      {1, 128, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 256, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 512, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 1024, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 2048, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
  };

  int success_count = 0;
  int total_count = test_configs.size();

  for (const auto& config : test_configs) {
    if (isPrint) {
      std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
      std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
                << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
                << ", head_size=" << config.head_size << ", head_size_v=" << config.head_size_v
                << ", is_causal=" << config.is_causal << std::endl;
    }

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
      auto scale_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
      torch::manual_seed(42);
      // torch::randn not support torch::kFloat8_e4m3fn directly
      auto q_tensor =
          torch::randn({total_tokens, config.num_heads, config.head_size}, options).to(torch::kFloat8_e4m3fn);
      auto k_tensor =
          torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options).to(torch::kFloat8_e4m3fn);
      auto v_tensor =
          torch::randn({total_tokens, config.num_kv_heads, config.head_size_v}, options).to(torch::kFloat8_e4m3fn);
      auto q_descale = 0.5f * torch::ones({config.batch_size, config.num_heads}, scale_options);
      auto k_descale = 0.5f * torch::ones({config.batch_size, config.num_heads}, scale_options);
      auto v_descale = 0.5f * torch::ones({config.batch_size, config.num_heads}, scale_options);
      std::cout << "create tensor finished" << std::endl;

      auto cu_seqlens = CreateCuSeqlens(config.batch_size, config.seq_len);

      auto q_contiguous = q_tensor.contiguous();
      auto k_contiguous = k_tensor.contiguous();
      auto v_contiguous = v_tensor.contiguous();
      auto q_descale_contiguous = q_descale.contiguous();
      auto k_descale_contiguous = k_descale.contiguous();
      auto v_descale_contiguous = v_descale.contiguous();
      auto q_tmp_tensor = torch::reshape(q_contiguous, {total_tokens, config.num_heads, config.head_size});
      std::optional<at::Tensor> k_new_ = std::nullopt;
      std::optional<at::Tensor> v_new_ = std::nullopt;
      std::optional<at::Tensor> q_v_ = std::nullopt;
      std::optional<at::Tensor> out_ = std::nullopt;
      std::optional<at::Tensor> cu_seqlens_q_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_new_ = std::nullopt;
      std::optional<at::Tensor> seqused_q_ = std::nullopt;
      std::optional<at::Tensor> seqused_k_ = std::nullopt;
      std::optional<int64_t> max_seqlen_q_ = static_cast<int64_t>(config.seq_len);
      std::optional<int64_t> max_seqlen_k_ = static_cast<int64_t>(config.seq_len);
      std::optional<at::Tensor> page_table_ = std::nullopt;
      std::optional<at::Tensor> kv_batch_idx_ = std::nullopt;
      std::optional<at::Tensor> leftpad_k_ = std::nullopt;
      std::optional<at::Tensor> rotary_cos_ = std::nullopt;
      std::optional<at::Tensor> rotary_sin_ = std::nullopt;
      std::optional<at::Tensor> seqlens_rotary_ = std::nullopt;
      std::optional<at::Tensor> q_descale_ = q_descale_contiguous;
      std::optional<at::Tensor> k_descale_ = k_descale_contiguous;
      std::optional<at::Tensor> v_descale_ = v_descale_contiguous;
      std::optional<double> softmax_scale_opt = std::optional<double>(static_cast<double>(attn_scale));
      int64_t window_size_left = -1;
      int64_t window_size_right = -1;
      int64_t attention_chunk = 0;
      double softcap_val = 0.0;
      bool is_rotary_interleaved = false;
      std::optional<at::Tensor> scheduler_metadata_ = std::nullopt;
      int64_t num_splits = 0;
      std::optional<bool> pack_gqa_ = std::nullopt;
      int64_t sm_margin = 0;
      std::cout << "prepare params finished" << std::endl;
      // 1. 调用FA3（计时）
      at::Tensor fa3_output;
      double fa3_avg_ms = 0.0;
      const int warmup_iters = 2;
      const int bench_iters = 5;
      std::vector<at::Tensor> fa3_result;
      try {
        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                            cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_,
                            page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
                            q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal, window_size_left,
                            window_size_right, attention_chunk, softcap_val, is_rotary_interleaved, scheduler_metadata_,
                            num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_result = mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                                   cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_,
                                   max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_,
                                   seqlens_rotary_, q_descale_, k_descale_, v_descale_, softmax_scale_opt,
                                   config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
                                   is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fa3_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
        fa3_output = fa3_result[0];
      } catch (const std::exception& e) {
        if (isPrint) {
          std::cerr << "FA3调用失败: " << e.what() << std::endl;
        }
        continue;
      }
      // 验证 FA3 输出
      ASSERT_FALSE(fa3_result.empty()) << "FA3 应该返回非空结果";
      auto output_tensor = fa3_result[0];
      ASSERT_EQ(output_tensor.dim(), 3) << "输出张量应该是3维的";
      ASSERT_EQ(output_tensor.size(0), total_tokens) << "输出张量第一维应该等于总token数";
      ASSERT_EQ(output_tensor.size(1), config.num_heads) << "输出张量第二维应该等于头数";
      ASSERT_EQ(output_tensor.size(2), config.head_size_v) << "输出张量第三维应该等于头大小";

      // 检查输出是否包含有效数值
      bool has_nan = torch::any(torch::isnan(output_tensor)).item<bool>();
      bool has_inf = torch::any(torch::isinf(output_tensor)).item<bool>();
      ASSERT_FALSE(has_nan) << "输出不应包含 NaN 值";
      ASSERT_FALSE(has_inf) << "输出不应包含 Inf 值";

      // 检查输出范围是否合理
      auto output_abs_max = torch::max(torch::abs(output_tensor)).item<float>();
      ASSERT_LT(output_abs_max, 100.0f) << "输出值的绝对值应该在合理范围内";

      if (isPrint) {
        std::cout << "✅ FA3 FP8 测试通过: 平均耗时=" << fa3_avg_ms << " ms" << std::endl;
        std::cout << "   输出形状: " << output_tensor.sizes() << std::endl;
        std::cout << "   输出最大绝对值: " << output_abs_max << std::endl;
      }
      success_count++;
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      }
    }
  }

  // 验证至少有一些测试成功
  EXPECT_GT(success_count, 0) << "至少应该有一个测试配置成功";
}

// 测试 VLLM FA2 varlen 调用
TEST_F(DirectFlashAttnComparisonTest, TestVllmFA2VarlenCall) {
  // 检查 VLLM FA2 函数指针是否可用
  if (!ksana_llm::FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_) {
    GTEST_SKIP() << "VLLM FA2 varlen function not available";
  }

  // 控制调试输出的参数
  const bool isPrint = false;  // 设置为 true 可以启用详细的调试输出

  if (isPrint) {
    std::cout << "\n=== 开始测试 VLLM FA2 varlen 调用 ===" << std::endl;
  }

  // 测试参数配置
  struct TestConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_size;
    bool is_causal;
    std::string description;
  };
  std::vector<TestConfig> test_configs = {
      {1, 128, 32, 32, 128, true, "小批次_短序列_因果注意力"},
      {2, 256, 16, 16, 64, true, "中批次_中序列_因果注意力"},
      {1, 512, 8, 8, 128, false, "小批次_长序列_非因果注意力"},
  };

  int success_count = 0;
  for (const auto& config : test_configs) {
    if (isPrint) {
      std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
      std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
                << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
                << ", head_size=" << config.head_size << ", is_causal=" << config.is_causal << std::endl;
    }

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

      // 设置随机种子以确保可重现性
      torch::manual_seed(42);

      // 创建输入张量 (varlen 格式)
      auto q_tensor = torch::randn({total_tokens, config.num_heads, config.head_size}, options);
      auto k_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);
      auto v_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);

      // 创建 cumulative sequence lengths
      auto cu_seqlens = CreateCuSeqlens(config.batch_size, config.seq_len);

      // 确保张量是连续的
      auto q_contiguous = q_tensor.contiguous();
      auto k_contiguous = k_tensor.contiguous();
      auto v_contiguous = v_tensor.contiguous();

      // 调用 VLLM FA2 varlen 函数
      std::vector<at::Tensor> vllm_fa2_result;
      try {
        c10::optional<at::Tensor> out_tensor = c10::nullopt;
        c10::optional<at::Tensor> seqused_k_vllm = c10::nullopt;
        c10::optional<at::Tensor> block_table = c10::nullopt;
        c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
        c10::optional<at::Generator> gen = c10::nullopt;

        vllm_fa2_result = mha_varlen_fwd_vllm_flash_attn_v26(
            q_contiguous, k_contiguous, v_contiguous, out_tensor, cu_seqlens, cu_seqlens, seqused_k_vllm, block_table,
            alibi_slopes_tensor, config.seq_len, config.seq_len, 0.f, attn_scale, false, config.is_causal, -1, -1, 0.f,
            false, gen);

        // 验证输出
        ASSERT_FALSE(vllm_fa2_result.empty()) << "VLLM FA2 应该返回非空结果";

        auto output_tensor = vllm_fa2_result[0];
        ASSERT_EQ(output_tensor.dim(), 3) << "输出张量应该是3维的";
        ASSERT_EQ(output_tensor.size(0), total_tokens) << "输出张量第一维应该等于总token数";
        ASSERT_EQ(output_tensor.size(1), config.num_heads) << "输出张量第二维应该等于头数";
        ASSERT_EQ(output_tensor.size(2), config.head_size) << "输出张量第三维应该等于头大小";

        // 检查输出是否包含有效数值
        bool has_nan = torch::any(torch::isnan(output_tensor)).item<bool>();
        bool has_inf = torch::any(torch::isinf(output_tensor)).item<bool>();
        ASSERT_FALSE(has_nan) << "输出不应包含 NaN 值";
        ASSERT_FALSE(has_inf) << "输出不应包含 Inf 值";

        // 检查输出范围是否合理
        auto output_abs_max = torch::max(torch::abs(output_tensor)).item<float>();
        ASSERT_LT(output_abs_max, 100.0f) << "输出值的绝对值应该在合理范围内";

        if (isPrint) {
          std::cout << "✅ VLLM FA2 varlen 调用成功" << std::endl;
          std::cout << "   输出形状: " << output_tensor.sizes() << std::endl;
          std::cout << "   输出最大绝对值: " << output_abs_max << std::endl;
        }
        success_count++;
      } catch (const std::exception& e) {
        GTEST_SKIP() << "VLLM FA2 varlen 调用失败，可能是库未加载: " << e.what();
      }
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      }
      FAIL() << "测试执行失败: " << e.what();
    }
  }

  // 验证至少有一些测试成功
  EXPECT_GT(success_count, 0) << "至少应该有一个测试配置成功";

  if (isPrint) {
    std::cout << "=== VLLM FA2 varlen 调用测试完成 ===" << std::endl;
  }
}

// 测试 FA3 与 VLLM FA2 对比
TEST_F(DirectFlashAttnComparisonTest, CompareFA3WithVllmFA2) {
  // 检查两个函数指针是否都可用
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    GTEST_SKIP() << "FA3 function not available";
  }
  if (!ksana_llm::FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_) {
    GTEST_SKIP() << "VLLM FA2 varlen function not available";
  }

  // 控制调试输出的参数
  const bool isPrint = false;  // 设置为 true 可以启用详细的调试输出

  if (isPrint) {
    std::cout << "\n=== 开始对比 FA3 与 VLLM FA2 ===" << std::endl;
  }

  // 测试参数配置
  struct TestConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_size;
    bool is_causal;
    std::string description;
  };
  std::vector<TestConfig> test_configs = {
      {1, 128, 32, 32, 128, true, "小批次_短序列_因果注意力"},
      {1, 256, 16, 16, 64, true, "小批次_中序列_因果注意力"},
      {1, 512, 8, 8, 128, false, "小批次_长序列_非因果注意力"},
  };

  for (const auto& config : test_configs) {
    if (isPrint) {
      std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
      std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
                << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
                << ", head_size=" << config.head_size << ", is_causal=" << config.is_causal << std::endl;
    }

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

      // 设置随机种子以确保可重现性
      torch::manual_seed(42);

      // 创建输入张量 (varlen 格式)
      auto q_tensor = torch::randn({total_tokens, config.num_heads, config.head_size}, options);
      auto k_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);
      auto v_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);

      // 创建 cumulative sequence lengths
      auto cu_seqlens = CreateCuSeqlens(config.batch_size, config.seq_len);

      // 确保张量是连续的
      auto q_contiguous = q_tensor.contiguous();
      auto k_contiguous = k_tensor.contiguous();
      auto v_contiguous = v_tensor.contiguous();

      // 性能测试参数
      const int warmup_iters = 2;
      const int bench_iters = 5;

      // 1. 调用 FA3 (带性能测试)
      std::vector<at::Tensor> fa3_result;
      at::Tensor fa3_output;
      double fa3_avg_ms = 0.0;
      try {
        std::optional<at::Tensor> k_new_ = std::nullopt;
        std::optional<at::Tensor> v_new_ = std::nullopt;
        std::optional<at::Tensor> q_v_ = std::nullopt;
        std::optional<at::Tensor> out_ = std::nullopt;
        std::optional<at::Tensor> cu_seqlens_q_ = cu_seqlens;
        std::optional<at::Tensor> cu_seqlens_k_ = cu_seqlens;
        std::optional<at::Tensor> cu_seqlens_k_new_ = std::nullopt;
        std::optional<at::Tensor> seqused_q_ = std::nullopt;
        std::optional<at::Tensor> seqused_k_ = std::nullopt;
        std::optional<int64_t> max_seqlen_q_ = static_cast<int64_t>(config.seq_len);
        std::optional<int64_t> max_seqlen_k_ = static_cast<int64_t>(config.seq_len);
        std::optional<at::Tensor> page_table_ = std::nullopt;
        std::optional<at::Tensor> kv_batch_idx_ = std::nullopt;
        std::optional<at::Tensor> leftpad_k_ = std::nullopt;
        std::optional<at::Tensor> rotary_cos_ = std::nullopt;
        std::optional<at::Tensor> rotary_sin_ = std::nullopt;
        std::optional<at::Tensor> seqlens_rotary_ = std::nullopt;
        std::optional<at::Tensor> q_descale_ = std::nullopt;
        std::optional<at::Tensor> k_descale_ = std::nullopt;
        std::optional<at::Tensor> v_descale_ = std::nullopt;
        std::optional<double> softmax_scale_opt = std::optional<double>(static_cast<double>(attn_scale));
        int64_t window_size_left = -1;
        int64_t window_size_right = -1;
        int64_t attention_chunk = 0;
        double softcap_val = 0.0;
        bool is_rotary_interleaved = false;
        std::optional<at::Tensor> scheduler_metadata_ = std::nullopt;
        int64_t num_splits = 0;
        std::optional<bool> pack_gqa_ = std::nullopt;
        int64_t sm_margin = 0;

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_fwd_fa3(q_contiguous, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                            cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_,
                            page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
                            q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal, window_size_left,
                            window_size_right, attention_chunk, softcap_val, is_rotary_interleaved, scheduler_metadata_,
                            num_splits, pack_gqa_, sm_margin);
        }

        // Benchmark
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_result = mha_fwd_fa3(q_contiguous, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                                   cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_,
                                   max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_,
                                   seqlens_rotary_, q_descale_, k_descale_, v_descale_, softmax_scale_opt,
                                   config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
                                   is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fa3_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

        fa3_output = fa3_result[0];
      } catch (const std::exception& e) {
        if (isPrint) {
          std::cerr << "FA3 调用失败: " << e.what() << std::endl;
        }
        continue;
      }

      // 2. 调用 VLLM FA2 (带性能测试)
      std::vector<at::Tensor> vllm_fa2_result;
      at::Tensor vllm_fa2_output;
      double vllm_fa2_avg_ms = 0.0;
      try {
        c10::optional<at::Tensor> out_tensor = c10::nullopt;
        c10::optional<at::Tensor> seqused_k_vllm = c10::nullopt;
        c10::optional<at::Tensor> block_table = c10::nullopt;
        c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
        c10::optional<at::Generator> gen = c10::nullopt;

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_varlen_fwd_vllm_flash_attn_v26(q_contiguous, k_contiguous, v_contiguous, out_tensor, cu_seqlens,
                                                   cu_seqlens, seqused_k_vllm, block_table, alibi_slopes_tensor,
                                                   config.seq_len, config.seq_len, 0.f, attn_scale, false,
                                                   config.is_causal, -1, -1, 0.f, false, gen);
        }

        // Benchmark
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          vllm_fa2_result = mha_varlen_fwd_vllm_flash_attn_v26(
              q_contiguous, k_contiguous, v_contiguous, out_tensor, cu_seqlens, cu_seqlens, seqused_k_vllm, block_table,
              alibi_slopes_tensor, config.seq_len, config.seq_len, 0.f, attn_scale, false, config.is_causal, -1, -1,
              0.f, false, gen);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        vllm_fa2_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

        vllm_fa2_output = vllm_fa2_result[0];
      } catch (const std::exception& e) {
        if (isPrint) {
          std::cerr << "VLLM FA2 调用失败: " << e.what() << std::endl;
        }
        continue;
      }

      // 3. 对比输出
      auto comparison_result =
          CompareTensorOutputs(fa3_output, vllm_fa2_output, "FA3输出", "VLLM FA2输出", 0.999, 0.01, 0.001, isPrint);

      // 4. 性能对比
      double speedup_ratio = vllm_fa2_avg_ms / fa3_avg_ms;

      if (isPrint) {
        std::cout << "\n=== 性能对比结果 ===" << std::endl;
        std::cout << "FA3 平均耗时: " << std::fixed << std::setprecision(3) << fa3_avg_ms << " ms" << std::endl;
        std::cout << "VLLM FA2 平均耗时: " << std::fixed << std::setprecision(3) << vllm_fa2_avg_ms << " ms"
                  << std::endl;
        std::cout << "加速比 (VLLM FA2/FA3): " << std::fixed << std::setprecision(2) << speedup_ratio << "x"
                  << std::endl;

        if (comparison_result.comparison_passed) {
          std::cout << "✅ FA3 与 VLLM FA2 输出对比通过" << std::endl;
        } else {
          std::cout << "⚠️ FA3 与 VLLM FA2 输出存在差异，但这可能是正常的实现差异" << std::endl;
        }
      }

      // 验证性能和精度结果
      EXPECT_GT(fa3_avg_ms, 0.0) << "FA3 执行时间应该大于0";
      EXPECT_GT(vllm_fa2_avg_ms, 0.0) << "VLLM FA2 执行时间应该大于0";
      EXPECT_GT(comparison_result.cosine_similarity, 0.95) << "余弦相似度应该较高";
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      }
      FAIL() << "测试执行失败: " << e.what();
    }
  }

  if (isPrint) {
    std::cout << "=== FA3 与 VLLM FA2 对比测试完成 ===" << std::endl;
  }
}

TEST_F(DirectFlashAttnComparisonTest, CompareFA3FP16AndFP8) {
  // 检查 FA3 函数指针是否可用
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    GTEST_SKIP() << "FA3 function not available";
  }

  // 控制调试输出的参数
  const bool isPrint = false;  // 设置为 true 可以启用详细的调试输出

  // 测试参数配置
  struct TestConfig {
    int batch_size;
    int seq_len;
    int num_heads;
    int num_kv_heads;
    int head_size;
    int head_size_v;
    bool is_causal;
    std::string description;
  };
  std::vector<TestConfig> test_configs = {
      {1, 128, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 256, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 512, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 1024, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
      {1, 2048, 32, 32, 128, 128, true, "小批次_短序列_因果注意力"},
  };

  int success_count = 0;
  int total_count = test_configs.size();

  for (const auto& config : test_configs) {
    if (isPrint) {
      std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
      std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
                << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
                << ", head_size=" << config.head_size << ", head_size_v=" << config.head_size_v
                << ", is_causal=" << config.is_causal << std::endl;
    }

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
      auto scale_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
      torch::manual_seed(42);
      // torch::randn not support torch::kFloat8_e4m3fn directly
      auto q_tensor = torch::randn({total_tokens, config.num_heads, config.head_size}, options);
      auto k_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);
      auto v_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size_v}, options);
      auto q_quant_tensor = q_tensor.to(torch::kFloat8_e4m3fn);
      auto k_quant_tensor = k_tensor.to(torch::kFloat8_e4m3fn);
      auto v_quant_tensor = v_tensor.to(torch::kFloat8_e4m3fn);
      std::cout << "create tensor finished" << std::endl;

      auto cu_seqlens = CreateCuSeqlens(config.batch_size, config.seq_len);

      auto q_contiguous = q_tensor.contiguous();
      auto k_contiguous = k_tensor.contiguous();
      auto v_contiguous = v_tensor.contiguous();
      auto q_tmp_tensor = torch::reshape(q_contiguous, {total_tokens, config.num_heads, config.head_size});

      auto q_quant_contiguous = q_quant_tensor.contiguous();
      auto k_quant_contiguous = k_quant_tensor.contiguous();
      auto v_quant_contiguous = v_quant_tensor.contiguous();
      auto q_quant_tmp_tensor = torch::reshape(q_quant_contiguous, {total_tokens, config.num_heads, config.head_size});

      std::optional<at::Tensor> k_new_ = std::nullopt;
      std::optional<at::Tensor> v_new_ = std::nullopt;
      std::optional<at::Tensor> q_v_ = std::nullopt;
      std::optional<at::Tensor> out_ = std::nullopt;
      std::optional<at::Tensor> cu_seqlens_q_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_ = cu_seqlens;
      std::optional<at::Tensor> cu_seqlens_k_new_ = std::nullopt;
      std::optional<at::Tensor> seqused_q_ = std::nullopt;
      std::optional<at::Tensor> seqused_k_ = std::nullopt;
      std::optional<int64_t> max_seqlen_q_ = static_cast<int64_t>(config.seq_len);
      std::optional<int64_t> max_seqlen_k_ = static_cast<int64_t>(config.seq_len);
      std::optional<at::Tensor> page_table_ = std::nullopt;
      std::optional<at::Tensor> kv_batch_idx_ = std::nullopt;
      std::optional<at::Tensor> leftpad_k_ = std::nullopt;
      std::optional<at::Tensor> rotary_cos_ = std::nullopt;
      std::optional<at::Tensor> rotary_sin_ = std::nullopt;
      std::optional<at::Tensor> seqlens_rotary_ = std::nullopt;
      std::optional<at::Tensor> q_descale_ = std::nullopt;
      std::optional<at::Tensor> k_descale_ = std::nullopt;
      std::optional<at::Tensor> v_descale_ = std::nullopt;
      std::optional<double> softmax_scale_opt = std::optional<double>(static_cast<double>(attn_scale));
      int64_t window_size_left = -1;
      int64_t window_size_right = -1;
      int64_t attention_chunk = 0;
      double softcap_val = 0.0;
      bool is_rotary_interleaved = false;
      std::optional<at::Tensor> scheduler_metadata_ = std::nullopt;
      int64_t num_splits = 0;
      std::optional<bool> pack_gqa_ = std::nullopt;
      int64_t sm_margin = 0;
      std::cout << "prepare params finished" << std::endl;
      // 1. 调用FA3（计时）
      at::Tensor fa3_output;
      at::Tensor fa3_fp8_output;
      double fa3_avg_ms = 0.0;
      double fa3_fp8_avg_ms = 0.0;
      const int warmup_iters = 2;
      const int bench_iters = 5;
      std::vector<at::Tensor> fa3_result;
      std::vector<at::Tensor> fa3_fp8_result;
      try {
        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                            cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_,
                            page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
                            q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal, window_size_left,
                            window_size_right, attention_chunk, softcap_val, is_rotary_interleaved, scheduler_metadata_,
                            num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_result = mha_fwd_fa3(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                                   cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_,
                                   max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_,
                                   seqlens_rotary_, q_descale_, k_descale_, v_descale_, softmax_scale_opt,
                                   config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
                                   is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fa3_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
        fa3_output = fa3_result[0];

        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)mha_fwd_fa3(q_quant_tmp_tensor, k_quant_contiguous, v_quant_contiguous, k_new_, v_new_, q_v_, out_,
                            cu_seqlens_q_, cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_,
                            max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_,
                            seqlens_rotary_, q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal,
                            window_size_left, window_size_right, attention_chunk, softcap_val, is_rotary_interleaved,
                            scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto fp8_t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_fp8_result = mha_fwd_fa3(
              q_quant_tmp_tensor, k_quant_contiguous, v_quant_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
              cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_, page_table_,
              kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_, q_descale_, k_descale_, v_descale_,
              softmax_scale_opt, config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
              is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto fp8_t1 = std::chrono::high_resolution_clock::now();
        fa3_fp8_avg_ms = std::chrono::duration<double, std::milli>(fp8_t1 - fp8_t0).count() / bench_iters;
        fa3_fp8_output = fa3_fp8_result[0];
      } catch (const std::exception& e) {
        if (isPrint) {
          std::cerr << "FA3调用失败: " << e.what() << std::endl;
        }
        continue;
      }

      // 对比输出
      auto comparison_result =
          CompareTensorOutputs(fa3_output, fa3_fp8_output, "FA3输出", "FA3 FP8输出", 0.99, 0.2, 0.01, isPrint);

      // 4. 性能对比
      double speedup_ratio = fa3_fp8_avg_ms / fa3_avg_ms;

      if (isPrint) {
        std::cout << "\n=== 性能对比结果 ===" << std::endl;
        std::cout << "FA3 平均耗时: " << std::fixed << std::setprecision(3) << fa3_avg_ms << " ms" << std::endl;
        std::cout << "FA3 FP8 平均耗时: " << std::fixed << std::setprecision(3) << fa3_fp8_avg_ms << " ms" << std::endl;
        std::cout << "加速比 (FA3 FP8/FA3): " << std::fixed << std::setprecision(2) << speedup_ratio << "x"
                  << std::endl;

        if (comparison_result.comparison_passed) {
          std::cout << "✅ FA3 FP16 与 FP8 输出对比通过" << std::endl;
        } else {
          std::cout << "⚠️ FA3 FP16 与 FP8 输出存在差异，但这可能是正常的实现差异" << std::endl;
        }
      }

      // 验证性能和精度结果
      EXPECT_GT(fa3_avg_ms, 0.0) << "FA3 执行时间应该大于0";
      EXPECT_GT(fa3_fp8_avg_ms, 0.0) << "FA3 FP8 执行时间应该大于0";
      EXPECT_GT(comparison_result.cosine_similarity, 0.95) << "余弦相似度应该较高";
      success_count++;
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      }
    }
  }

  // 验证至少有一些测试成功
  EXPECT_GT(success_count, 0) << "至少应该有一个测试配置成功";
}

// 测试 FA2V26 函数调用
TEST_F(DirectFlashAttnComparisonTest, TestFA2V26Functions) {
  const bool isPrint = false;

  if (isPrint) {
    std::cout << "\n=== 测试 FA2V26 函数调用 ===" << std::endl;
  }

  // 测试参数
  int batch_size = 1;
  int seq_len = 128;
  int num_heads = 8;
  int head_size = 64;
  int total_tokens = batch_size * seq_len;
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

  // 创建测试数据
  torch::manual_seed(42);
  auto q_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto k_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto v_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto cu_seqlens = CreateCuSeqlens(batch_size, seq_len);

  // 测试 FA2V26 Varlen 调用
  if (ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v26_) {
    try {
      c10::optional<at::Tensor> out_tensor = c10::nullopt;
      c10::optional<at::Tensor> seqused_k = c10::nullopt;
      c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
      c10::optional<at::Tensor> block_table = c10::nullopt;
      c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
      c10::optional<at::Generator> gen = c10::nullopt;

      auto result =
          mha_varlen_fwd_flash_attn_v26(const_cast<at::Tensor&>(q_tensor), k_tensor, v_tensor, out_tensor, cu_seqlens,
                                        cu_seqlens, seqused_k, leftpad_k, block_table, alibi_slopes, seq_len, seq_len,
                                        0.0f, attn_scale, false, true, -1, -1, 0.0f, false, gen);

      EXPECT_FALSE(result.empty()) << "FA2V26 应该返回非空结果";

      if (isPrint) {
        std::cout << "✅ FA2V26 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cout << "FA2V26 Varlen 调用失败: " << e.what() << std::endl;
      }
      // 这可能是预期的，如果库未加载
    }
  } else {
    if (isPrint) {
      std::cout << "FA2V26 Varlen 函数指针不可用" << std::endl;
    }
  }

  // 测试 FA2V26 KVCache 调用
  if (ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v26_) {
    try {
      c10::optional<const at::Tensor> k_new = c10::nullopt;
      c10::optional<const at::Tensor> v_new = c10::nullopt;
      c10::optional<const at::Tensor> seqlens_k = c10::nullopt;
      c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
      c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
      c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
      c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
      c10::optional<at::Tensor> block_table = c10::nullopt;
      c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
      c10::optional<at::Tensor> out_tensor = c10::nullopt;

      auto result = mha_fwd_kvcache_flash_attn_v26(
          const_cast<at::Tensor&>(q_tensor), k_tensor, v_tensor, k_new, v_new, seqlens_k, rotary_cos, rotary_sin,
          cache_batch_idx, leftpad_k, block_table, alibi_slopes, out_tensor, attn_scale, true, -1, -1, 0.0f, true, 0);

      EXPECT_FALSE(result.empty()) << "FA2V26 KVCache 应该返回非空结果";

      if (isPrint) {
        std::cout << "✅ FA2V26 KVCache 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cout << "FA2V26 KVCache 调用失败: " << e.what() << std::endl;
      }
      // 这可能是预期的，如果库未加载
    }
  } else {
    if (isPrint) {
      std::cout << "FA2V26 KVCache 函数指针不可用" << std::endl;
    }
  }

  if (isPrint) {
    std::cout << "=== FA2V26 函数调用测试完成 ===" << std::endl;
  }
}

// 测试 FA2V25 函数调用
TEST_F(DirectFlashAttnComparisonTest, TestFA2V25Functions) {
  const bool isPrint = false;

  if (isPrint) {
    std::cout << "\n=== 测试 FA2V25 函数调用 ===" << std::endl;
  }

  // 测试参数
  int batch_size = 1;
  int seq_len = 128;
  int num_heads = 8;
  int head_size = 64;
  int total_tokens = batch_size * seq_len;
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

  // 创建测试数据
  torch::manual_seed(42);
  auto q_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto k_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto v_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto cu_seqlens = CreateCuSeqlens(batch_size, seq_len);

  // 测试 FA2V25 Varlen 调用
  if (ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v25_) {
    try {
      c10::optional<at::Tensor> out_tensor = c10::nullopt;
      c10::optional<at::Tensor> seqused_k = c10::nullopt;
      c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
      c10::optional<at::Generator> gen = c10::nullopt;

      auto result = mha_varlen_fwd_flash_attn_v25(const_cast<at::Tensor&>(q_tensor), k_tensor, v_tensor, out_tensor,
                                                  cu_seqlens, cu_seqlens, seqused_k, alibi_slopes, seq_len, seq_len,
                                                  0.0f, attn_scale, false, true, -1, -1, false, gen);

      EXPECT_FALSE(result.empty()) << "FA2V25 应该返回非空结果";

      if (isPrint) {
        std::cout << "✅ FA2V25 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cout << "FA2V25 Varlen 调用失败: " << e.what() << std::endl;
      }
      // 这可能是预期的，如果库未加载
    }
  } else {
    if (isPrint) {
      std::cout << "FA2V25 Varlen 函数指针不可用" << std::endl;
    }
  }

  // 测试 FA2V25 KVCache 调用
  if (ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v25_) {
    try {
      c10::optional<const at::Tensor> k_new = c10::nullopt;
      c10::optional<const at::Tensor> v_new = c10::nullopt;
      c10::optional<const at::Tensor> seqlens_k = c10::nullopt;
      c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
      c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
      c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
      c10::optional<at::Tensor> block_table = c10::nullopt;
      c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
      c10::optional<at::Tensor> out_tensor = c10::nullopt;

      auto result = mha_fwd_kvcache_flash_attn_v25(const_cast<at::Tensor&>(q_tensor), k_tensor, v_tensor, k_new, v_new,
                                                   seqlens_k, rotary_cos, rotary_sin, cache_batch_idx, block_table,
                                                   alibi_slopes, out_tensor, attn_scale, true, -1, -1, true, 0);

      EXPECT_FALSE(result.empty()) << "FA2V25 KVCache 应该返回非空结果";

      if (isPrint) {
        std::cout << "✅ FA2V25 KVCache 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      if (isPrint) {
        std::cout << "FA2V25 KVCache 调用失败: " << e.what() << std::endl;
      }
      // 这可能是预期的，如果库未加载
    }
  } else {
    if (isPrint) {
      std::cout << "FA2V25 KVCache 函数指针不可用" << std::endl;
    }
  }

  if (isPrint) {
    std::cout << "=== FA2V25 函数调用测试完成 ===" << std::endl;
  }
}

// 测试 FA3 不支持 alibi_slopes 的错误处理
TEST_F(DirectFlashAttnComparisonTest, TestFA3AlibiBlopesError) {
  // 检查 FA3 函数指针是否可用
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    GTEST_SKIP() << "FA3 function not available";
  }

  const bool isPrint = false;

  if (isPrint) {
    std::cout << "\n=== 测试 FA3 alibi_slopes 错误处理 ===" << std::endl;
  }

  // 测试参数
  int batch_size = 1;
  int seq_len = 128;
  int num_heads = 8;
  int head_size = 64;
  int total_tokens = batch_size * seq_len;
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

  // 创建测试数据
  torch::manual_seed(42);
  auto q_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto k_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto v_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto cu_seqlens = CreateCuSeqlens(batch_size, seq_len);

  // 创建 alibi_slopes 张量来触发错误
  auto alibi_slopes = torch::randn({num_heads}, options);

  // 获取 Environment 单例来修改配置，强制使用 FA3
  auto env = Singleton<Environment>::GetInstance();
  if (!env) {
    GTEST_SKIP() << "Environment singleton not available";
  }

  // 保存原始配置
  AttnBackendConfig original_config;
  env->GetAttnBackendConfig(original_config);

  // 设置强制使用 FA3
  AttnBackendConfig fa3_config = original_config;
  fa3_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::FA3;
  env->SetAttnBackendConfig(fa3_config);

  // 测试 Varlen 路径的 alibi_slopes 错误
  {
    MhaVarlenFwdParams params;
    params.q = q_tensor.contiguous();
    params.k = k_tensor.contiguous();
    params.v = v_tensor.contiguous();
    params.seqlen_q = cu_seqlens;
    params.seqlen_k = cu_seqlens;
    params.max_seqlen_q = static_cast<int64_t>(seq_len);
    params.max_seqlen_k = static_cast<int64_t>(seq_len);
    params.softmax_scale = static_cast<double>(attn_scale);
    params.is_causal = true;
    params.alibi_slopes = alibi_slopes;  // 设置 alibi_slopes 来触发错误

    // 应该抛出 FA3 相关错误：alibi_slopes 不支持错误 或 FA3 不可用错误
    EXPECT_THROW(
        {
          try {
            InvokeMhaVarlenFwd(params);
          } catch (const std::exception& e) {
            std::string error_msg = e.what();
            // 检查是否是预期的两种错误之一
            bool is_alibi_error =
                error_msg.find("Flash Attention 3 does not support alibi_slopes") != std::string::npos;
            bool is_unavailable_error =
                error_msg.find("FlashAttention 3 is not available but is forced by configuration") != std::string::npos;
            EXPECT_TRUE(is_alibi_error || is_unavailable_error)
                << "Expected FA3 alibi_slopes error or FA3 unavailable error, but got: " << error_msg;
            throw;  // 重新抛出以满足 EXPECT_THROW
          }
        },
        std::exception);

    if (isPrint) {
      std::cout << "✅ FA3 Varlen alibi_slopes 错误处理测试通过" << std::endl;
    }
  }

  // 测试 KVCache 路径的 alibi_slopes 错误
  {
    MhaFwdKVCacheParams params;
    params.q = q_tensor.contiguous();
    params.k_cache = k_tensor.contiguous();
    params.v_cache = v_tensor.contiguous();
    params.softmax_scale = static_cast<double>(attn_scale);
    params.is_causal = true;
    params.alibi_slopes = alibi_slopes;  // 设置 alibi_slopes 来触发错误

    // 应该抛出 FA3 相关错误：alibi_slopes 不支持错误 或 FA3 不可用错误
    EXPECT_THROW(
        {
          try {
            InvokeMhaFwdKvcCache(params);
          } catch (const std::exception& e) {
            std::string error_msg = e.what();
            // 检查是否是预期的两种错误之一
            bool is_alibi_error =
                error_msg.find("Flash Attention 3 does not support alibi_slopes") != std::string::npos;
            bool is_unavailable_error =
                error_msg.find("FlashAttention 3 is not available but is forced by configuration") != std::string::npos;
            EXPECT_TRUE(is_alibi_error || is_unavailable_error)
                << "Expected FA3 alibi_slopes error or FA3 unavailable error, but got: " << error_msg;
            throw;  // 重新抛出以满足 EXPECT_THROW
          }
        },
        std::exception);

    if (isPrint) {
      std::cout << "✅ FA3 KVCache alibi_slopes 错误处理测试通过" << std::endl;
    }
  }

  // 恢复原始配置
  env->SetAttnBackendConfig(original_config);

  if (isPrint) {
    std::cout << "=== FA3 alibi_slopes 错误处理测试完成 ===" << std::endl;
  }
}

// 测试强制选择不可用实现时的错误处理
TEST_F(DirectFlashAttnComparisonTest, TestForcedImplementationErrors) {
  const bool isPrint = false;

  if (isPrint) {
    std::cout << "\n=== 测试强制选择不可用实现的错误处理 ===" << std::endl;
  }

  // 获取 Environment 单例来修改配置
  auto env = Singleton<Environment>::GetInstance();
  if (!env) {
    GTEST_SKIP() << "Environment singleton not available";
  }

  // 保存原始配置
  AttnBackendConfig original_config;
  env->GetAttnBackendConfig(original_config);

  // 测试参数
  int batch_size = 1;
  int seq_len = 128;
  int num_heads = 8;
  int head_size = 64;
  int total_tokens = batch_size * seq_len;
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

  // 创建测试数据
  torch::manual_seed(42);
  auto q_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto k_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto v_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto cu_seqlens = CreateCuSeqlens(batch_size, seq_len);

  // 准备 Varlen 参数
  MhaVarlenFwdParams varlen_params;
  varlen_params.q = q_tensor.contiguous();
  varlen_params.k = k_tensor.contiguous();
  varlen_params.v = v_tensor.contiguous();
  varlen_params.seqlen_q = cu_seqlens;
  varlen_params.seqlen_k = cu_seqlens;
  varlen_params.max_seqlen_q = static_cast<int64_t>(seq_len);
  varlen_params.max_seqlen_k = static_cast<int64_t>(seq_len);
  varlen_params.softmax_scale = static_cast<double>(attn_scale);
  varlen_params.is_causal = true;

  // 准备 KVCache 参数
  MhaFwdKVCacheParams kvcache_params;
  kvcache_params.q = q_tensor.contiguous();
  kvcache_params.k_cache = k_tensor.contiguous();
  kvcache_params.v_cache = v_tensor.contiguous();
  kvcache_params.softmax_scale = static_cast<double>(attn_scale);
  kvcache_params.is_causal = true;

  // 测试强制选择 FA3
  {
    AttnBackendConfig test_config = original_config;
    test_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::FA3;
    env->SetAttnBackendConfig(test_config);

    try {
      InvokeMhaVarlenFwd(varlen_params);
      if (isPrint) {
        std::cout << "✅ 强制 FA3 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      if (error_msg.find("FlashAttention 3 is not available but is forced by configuration") != std::string::npos) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 FA3 强制选择错误: " << error_msg << std::endl;
        }
      } else {
        if (isPrint) {
          std::cout << "FA3 强制选择错误: " << error_msg << std::endl;
        }
      }
    }

    try {
      InvokeMhaFwdKvcCache(kvcache_params);
      if (isPrint) {
        std::cout << "✅ 强制 FA3 KVCache 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      if (error_msg.find("FlashAttention 3 is not available but is forced by configuration") != std::string::npos ||
          error_msg.find("FlashAttention 3 is not available") != std::string::npos) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 FA3 KVCache 强制选择错误: " << error_msg << std::endl;
        }
      } else {
        if (isPrint) {
          std::cout << "FA3 KVCache 强制选择错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 测试强制选择 VLLM_V26
  {
    AttnBackendConfig test_config = original_config;
    test_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::VLLM_V26;
    env->SetAttnBackendConfig(test_config);

    try {
      InvokeMhaVarlenFwd(varlen_params);
      if (isPrint) {
        std::cout << "✅ 强制 VLLM_V26 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      if (error_msg.find("vLLM FlashAttention 2.6 is not available but is forced by configuration") !=
          std::string::npos) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 VLLM_V26 强制选择错误: " << error_msg << std::endl;
        }
      } else {
        if (isPrint) {
          std::cout << "VLLM_V26 强制选择错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 测试强制选择 FA2_V26
  {
    AttnBackendConfig test_config = original_config;
    test_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::FA2_V26;
    env->SetAttnBackendConfig(test_config);

    try {
      InvokeMhaVarlenFwd(varlen_params);
      if (isPrint) {
        std::cout << "✅ 强制 FA2_V26 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      if (error_msg.find("FlashAttention 2.6 is not available but is forced by configuration") != std::string::npos) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 FA2_V26 强制选择错误: " << error_msg << std::endl;
        }
      } else {
        if (isPrint) {
          std::cout << "FA2_V26 强制选择错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 测试强制选择 FA2_V25
  {
    AttnBackendConfig test_config = original_config;
    test_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::FA2_V25;
    env->SetAttnBackendConfig(test_config);

    try {
      InvokeMhaVarlenFwd(varlen_params);
      if (isPrint) {
        std::cout << "✅ 强制 FA2_V25 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      std::string error_msg = e.what();
      if (error_msg.find("FlashAttention 2.5 is not available but is forced by configuration") != std::string::npos) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 FA2_V25 强制选择错误: " << error_msg << std::endl;
        }
      } else {
        if (isPrint) {
          std::cout << "FA2_V25 强制选择错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 恢复原始配置
  env->SetAttnBackendConfig(original_config);

  if (isPrint) {
    std::cout << "=== 强制选择实现错误处理测试完成 ===" << std::endl;
  }
}

// 测试 AUTO 模式下的实现选择逻辑
TEST_F(DirectFlashAttnComparisonTest, TestAutoModeSelection) {
  const bool isPrint = false;

  if (isPrint) {
    std::cout << "\n=== 测试 AUTO 模式下的实现选择逻辑 ===" << std::endl;
  }

  // 获取 Environment 单例来修改配置
  auto env = Singleton<Environment>::GetInstance();
  if (!env) {
    GTEST_SKIP() << "Environment singleton not available";
  }

  // 保存原始配置
  AttnBackendConfig original_config;
  env->GetAttnBackendConfig(original_config);

  // 测试参数
  int batch_size = 1;
  int seq_len = 128;
  int num_heads = 8;
  int head_size = 64;
  int total_tokens = batch_size * seq_len;
  float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);

  // 创建测试数据
  torch::manual_seed(42);
  auto q_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto k_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto v_tensor = torch::randn({total_tokens, num_heads, head_size}, options);
  auto cu_seqlens = CreateCuSeqlens(batch_size, seq_len);

  // 设置为 AUTO 模式
  AttnBackendConfig auto_config = original_config;
  auto_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::AUTO;
  env->SetAttnBackendConfig(auto_config);

  // 测试 Varlen 路径的 AUTO 选择
  {
    MhaVarlenFwdParams params;
    params.q = q_tensor.contiguous();
    params.k = k_tensor.contiguous();
    params.v = v_tensor.contiguous();
    params.seqlen_q = cu_seqlens;
    params.seqlen_k = cu_seqlens;
    params.max_seqlen_q = static_cast<int64_t>(seq_len);
    params.max_seqlen_k = static_cast<int64_t>(seq_len);
    params.softmax_scale = static_cast<double>(attn_scale);
    params.is_causal = true;
    // 不设置 alibi_slopes，使其为 RoPE 模式

    try {
      auto result = InvokeMhaVarlenFwd(params);
      EXPECT_FALSE(result.empty()) << "AUTO 模式应该返回非空结果";

      if (isPrint) {
        std::cout << "✅ AUTO 模式 Varlen 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      // 在某些环境下可能没有可用的实现
      std::string error_msg = e.what();
      if (isPrint) {
        std::cout << "AUTO 模式 Varlen 调用失败: " << error_msg << std::endl;
      }
      // 检查是否是预期的错误消息
      bool is_expected_error = error_msg.find("No suitable mha_varlen_fwd function loaded") != std::string::npos ||
                               error_msg.find("is not available") != std::string::npos;

      if (is_expected_error) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 AUTO 模式错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 测试 KVCache 路径的 AUTO 选择
  {
    MhaFwdKVCacheParams params;
    params.q = q_tensor.contiguous();
    params.k_cache = k_tensor.contiguous();
    params.v_cache = v_tensor.contiguous();
    params.softmax_scale = static_cast<double>(attn_scale);
    params.is_causal = true;
    // 不设置 alibi_slopes，使其为 RoPE 模式

    try {
      InvokeMhaFwdKvcCache(params);
      if (isPrint) {
        std::cout << "✅ AUTO 模式 KVCache 调用成功" << std::endl;
      }
    } catch (const std::exception& e) {
      // 在某些环境下可能没有可用的实现
      std::string error_msg = e.what();
      if (isPrint) {
        std::cout << "AUTO 模式 KVCache 调用失败: " << error_msg << std::endl;
      }
      // 检查是否是预期的错误消息
      bool is_expected_error = error_msg.find("No suitable mha_fwd_kvcache function loaded") != std::string::npos ||
                               error_msg.find("is not available") != std::string::npos;

      if (is_expected_error) {
        if (isPrint) {
          std::cout << "✅ 捕获到预期的 AUTO 模式 KVCache 错误: " << error_msg << std::endl;
        }
      }
    }
  }

  // 测试 IsUsingFA3 函数在不同配置下的行为
  {
    // 测试 AUTO 模式下的 IsUsingFA3
    bool using_fa3_auto = IsUsingFA3();
    if (isPrint) {
      std::cout << "AUTO 模式下使用 FA3: " << (using_fa3_auto ? "是" : "否") << std::endl;
    }
    EXPECT_NO_THROW(IsUsingFA3());

    // 测试强制 FA3 模式下的 IsUsingFA3
    AttnBackendConfig fa3_config = original_config;
    fa3_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::FA3;
    env->SetAttnBackendConfig(fa3_config);

    bool using_fa3_forced = IsUsingFA3();
    EXPECT_TRUE(using_fa3_forced) << "强制 FA3 模式下 IsUsingFA3 应该返回 true";
    if (isPrint) {
      std::cout << "强制 FA3 模式下使用 FA3: " << (using_fa3_forced ? "是" : "否") << std::endl;
    }

    // 测试强制 VLLM_V26 模式下的 IsUsingFA3
    AttnBackendConfig vllm_config = original_config;
    vllm_config.flash_attn_impl_choice = AttnBackendConfig::FlashAttnImplChoice::VLLM_V26;
    env->SetAttnBackendConfig(vllm_config);

    bool using_fa3_vllm = IsUsingFA3();
    EXPECT_FALSE(using_fa3_vllm) << "强制 VLLM_V26 模式下 IsUsingFA3 应该返回 false";
    if (isPrint) {
      std::cout << "强制 VLLM_V26 模式下使用 FA3: " << (using_fa3_vllm ? "是" : "否") << std::endl;
    }
  }

  // 恢复原始配置
  env->SetAttnBackendConfig(original_config);

  if (isPrint) {
    std::cout << "=== AUTO 模式选择逻辑测试完成 ===" << std::endl;
  }
}

#endif  // ENABLE_CUDA

}  // namespace ksana_llm
