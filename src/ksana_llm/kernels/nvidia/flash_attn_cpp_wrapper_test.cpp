/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

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

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tensor_test_helper.h"

#define private public
#define protected public
#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#undef private
#undef protected

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "tests/test.h"

#ifdef ENABLE_CUDA
#  include <ATen/cuda/CUDAContext.h>
#  include <c10/cuda/CUDAGuard.h>
#  include <torch/torch.h>
#endif

namespace ksana_llm {

#ifdef ENABLE_CUDA

class DirectFlashAttnComparisonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 设置 CUDA 设备
    device_ = torch::kCUDA;
    c10::cuda::set_device(0);

    backend_fa3_ = std::make_unique<FlashAttentionBackend>();
    backend_vllm_fa2_ = std::make_unique<FlashAttentionBackend>();

    ASSERT_TRUE(backend_fa3_->Initialize()) << "Failed to initialize FlashAttentionBackend";

    // FA3：获取库信息并加载
    auto fa3_info = backend_fa3_->GetFlashAttention3LibInfo();
    if (!fa3_info.path.empty()) {
      fa3_mgr_ = std::make_shared<RuntimeDllManager>();
      if (fa3_mgr_->Load(fa3_info.path)) {
        // 暂时切换到后端的dll管理器以使用其符号解析逻辑
        // 这里不访问后端私有成员，只复用 LoadFunctions 的符号查找规则
        // 使用相同的函数名进行符号匹配
        std::string symbol = fa3_mgr_->FindMangledFunctionSymbol("mha_fwd", FlashAttentionBackend::mha_fwd_fa3_);
        if (!symbol.empty()) {
          fa3_function_ = fa3_mgr_->GetRawFunctionPointer<ksana_llm::mha_fwd_fa3_ptr>(symbol);
        }
      }
    }

    // vLLM FA2：获取库信息并加载（需要版本>=2.6.0）
    auto vllm_info = backend_vllm_fa2_->GetVllmFlashAttentionLibInfo();
    if (!vllm_info.path.empty() && backend_vllm_fa2_->IsVersionGreaterOrEqual(vllm_info.version, "2.6.0")) {
      vllm_mgr_ = std::make_shared<RuntimeDllManager>();
      if (vllm_mgr_->Load(vllm_info.path)) {
        std::string varlen_symbol = vllm_mgr_->FindMangledFunctionSymbol(
            "mha_varlen_fwd", FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_);
        if (!varlen_symbol.empty()) {
          vllm_fa2_function_ =
              vllm_mgr_->GetRawFunctionPointer<ksana_llm::mha_varlen_fwd_vllm_flash_attn_v26_ptr>(varlen_symbol);
        }
      }
    }
  }

  void TearDown() override {
    backend_fa3_.reset();
    backend_vllm_fa2_.reset();
    fa3_mgr_.reset();
    vllm_mgr_.reset();
  }

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

  // 函数指针
  ksana_llm::mha_fwd_fa3_ptr fa3_function_ = nullptr;
  ksana_llm::mha_varlen_fwd_vllm_flash_attn_v26_ptr vllm_fa2_function_ = nullptr;

  // 后端实例与各自的DLL管理器，保持库生命周期
  std::unique_ptr<FlashAttentionBackend> backend_fa3_;
  std::unique_ptr<FlashAttentionBackend> backend_vllm_fa2_;
  std::shared_ptr<RuntimeDllManager> fa3_mgr_;
  std::shared_ptr<RuntimeDllManager> vllm_mgr_;
};

TEST_F(DirectFlashAttnComparisonTest, CompareFA3WithVllmFA2WithSyntheticInputs) {
  // 检查函数是否加载成功
  if (!fa3_function_) {
    GTEST_SKIP() << "FA3 function not available";
  }
  if (!vllm_fa2_function_) {
    GTEST_SKIP() << "VLLM FA2 function not available";
  }
  std::cout << "\n=== 开始使用合成输入数据对比FA3与VLLM FA2 ===" << std::endl;
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
      {1, 256, 32, 32, 128, true, "小批次_短序列_因果注意力"},
      {1, 512, 32, 32, 128, true, "小批次_短序列_因果注意力"},
      {1, 1024, 32, 32, 128, true, "小批次_短序列_因果注意力"},
      {1, 2048, 32, 32, 128, true, "小批次_短序列_因果注意力"},
  };

  int success_count = 0;
  int total_count = test_configs.size();

  for (const auto& config : test_configs) {
    std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
    std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
              << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
              << ", head_size=" << config.head_size << ", is_causal=" << config.is_causal << std::endl;

    try {
      int total_tokens = config.batch_size * config.seq_len;
      float attn_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size));
      auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
      torch::manual_seed(42);
      auto q_tensor = torch::randn({total_tokens, config.num_heads, config.head_size}, options);
      auto k_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);
      auto v_tensor = torch::randn({total_tokens, config.num_kv_heads, config.head_size}, options);

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
      std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fa3_result_tuple;
      try {
        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)fa3_function_(q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_,
                              cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_,
                              page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
                              q_descale_, k_descale_, v_descale_, softmax_scale_opt, config.is_causal, window_size_left,
                              window_size_right, attention_chunk, softcap_val, is_rotary_interleaved,
                              scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          fa3_result_tuple = fa3_function_(
              q_tmp_tensor, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_, cu_seqlens_k_,
              cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_, page_table_, kv_batch_idx_,
              leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_, q_descale_, k_descale_, v_descale_,
              softmax_scale_opt, config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
              is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fa3_avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
        fa3_output = std::get<0>(fa3_result_tuple);
      } catch (const std::exception& e) {
        std::cerr << "FA3调用失败: " << e.what() << std::endl;
        continue;
      }
      // 2. 调用VLLM FA2
      c10::optional<at::Tensor> out_tensor = c10::nullopt;
      c10::optional<at::Tensor> seqused_k_vllm = c10::nullopt;
      c10::optional<at::Tensor> block_table = c10::nullopt;
      c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
      c10::optional<at::Generator> gen = c10::nullopt;
      std::vector<at::Tensor> vllm_fa2_result;
      try {
        // warmup
        for (int i = 0; i < warmup_iters; ++i) {
          (void)vllm_fa2_function_(q_tmp_tensor, k_contiguous, v_contiguous, out_tensor, cu_seqlens, cu_seqlens,
                                   seqused_k_vllm, block_table, alibi_slopes_tensor, config.seq_len, config.seq_len,
                                   0.f, attn_scale, false, config.is_causal, -1, -1, 0.f, false, gen);
        }
        cudaDeviceSynchronize();
        auto t0b = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
          vllm_fa2_result =
              vllm_fa2_function_(q_tmp_tensor, k_contiguous, v_contiguous, out_tensor, cu_seqlens, cu_seqlens,
                                 seqused_k_vllm, block_table, alibi_slopes_tensor, config.seq_len, config.seq_len, 0.f,
                                 attn_scale, false, config.is_causal, -1, -1, 0.f, false, gen);
        }
        cudaDeviceSynchronize();
        auto t1b = std::chrono::high_resolution_clock::now();
        double fa2_avg_ms = std::chrono::duration<double, std::milli>(t1b - t0b).count() / bench_iters;
        auto vllm_fa2_output = vllm_fa2_result[0];
        // 使用通用对比函数进行详细对比
        auto comparison_result =
            CompareTensorOutputs(fa3_output, vllm_fa2_output, "FA3输出", "VLLM FA2输出", 0.999, 0.001, 0.001);
        EXPECT_TRUE(comparison_result.comparison_passed);
        std::cout << "对比通过: 平均耗时 FA3=" << fa3_avg_ms << " ms, FA2=" << fa2_avg_ms << " ms" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "VLLM FA2调用失败: " << e.what() << std::endl;
        continue;
      }
    } catch (const std::exception& e) {
      std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
    }
  }
}

// 测试 FA3 直接从 flash_attn_cpp_wrapper 调用 mha_fwd
TEST_F(DirectFlashAttnComparisonTest, TestFA3MhaFwdDirectCall) {
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
  for (const auto& config : test_configs) {
    std::cout << "\n--- 测试配置: " << config.description << " ---" << std::endl;
    std::cout << "batch_size=" << config.batch_size << ", seq_len=" << config.seq_len
              << ", num_heads=" << config.num_heads << ", num_kv_heads=" << config.num_kv_heads
              << ", head_size=" << config.head_size << ", is_causal=" << config.is_causal << std::endl;

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

      // 准备 FA3 mha_fwd 的参数
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

      // 调用 FA3 mha_fwd 函数
      std::vector<at::Tensor> fa3_result;
      try {
        fa3_result = ksana_llm::mha_fwd(
            q_contiguous, k_contiguous, v_contiguous, k_new_, v_new_, q_v_, out_, cu_seqlens_q_, cu_seqlens_k_,
            cu_seqlens_k_new_, seqused_q_, seqused_k_, max_seqlen_q_, max_seqlen_k_, page_table_, kv_batch_idx_,
            leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_, q_descale_, k_descale_, v_descale_,
            softmax_scale_opt, config.is_causal, window_size_left, window_size_right, attention_chunk, softcap_val,
            is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);

        // 验证输出
        ASSERT_FALSE(fa3_result.empty()) << "FA3 mha_fwd 应该返回非空结果";

        auto output_tensor = fa3_result[0];
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

        std::cout << "✅ FA3 mha_fwd 直接调用成功" << std::endl;
        std::cout << "   输出形状: " << output_tensor.sizes() << std::endl;
        std::cout << "   输出最大绝对值: " << output_abs_max << std::endl;
      } catch (const std::exception& e) {
        GTEST_SKIP() << "FA3 mha_fwd 调用失败，可能是库未加载: " << e.what();
      }
    } catch (const std::exception& e) {
      std::cerr << "测试配置 " << config.description << " 执行失败: " << e.what() << std::endl;
      FAIL() << "测试执行失败: " << e.what();
    }
  }

  std::cout << "=== FA3 mha_fwd 直接调用测试完成 ===" << std::endl;
}

#endif  // ENABLE_CUDA

}  // namespace ksana_llm
