/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

#include <torch/torch.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/moeOp.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/utils.h"

using namespace llm_kernels::nvidia;
using namespace llm_kernels::nvidia::tensorrt_llm::dev;

template <typename T>
torch::ScalarType GetTorchDataType();
#define GET_TORCH_DATA_TYPE(T, TORCH_TYPE)  \
  template <>                               \
  torch::ScalarType GetTorchDataType<T>() { \
    return TORCH_TYPE;                      \
  }
GET_TORCH_DATA_TYPE(float, torch::kFloat32);
GET_TORCH_DATA_TYPE(half, torch::kFloat16);
GET_TORCH_DATA_TYPE(__nv_bfloat16, torch::kBFloat16);
GET_TORCH_DATA_TYPE(int32_t, torch::kInt32);
#undef GET_TORCH_DATA_TYPE

template <typename T>
std::string GetDataTypeName();
#define GET_DATA_TYPE_NAME(T, NAME)  \
  template <>                        \
  std::string GetDataTypeName<T>() { \
    return NAME;                     \
  }
GET_DATA_TYPE_NAME(half, " float16");
GET_DATA_TYPE_NAME(__nv_bfloat16, "bfloat16");
#undef GET_DATA_TYPE_NAME

namespace llm_kernels {
namespace nvidia {
namespace test {

// https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc3/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L551
struct CutlassFusedMoE {
  size_t ep_size = 1;
  size_t ep_rank = 0;

  size_t tp_size = 1;
  size_t tp_rank = 0;

  size_t cluster_size = 1;
  size_t cluster_rank = 0;

  size_t hidden_size;
  size_t intermediate_size;
  size_t num_experts;
  size_t top_k = 1;
  size_t scaling_group_size;

  size_t expert_size_per_partition;
  size_t intermediate_size_per_partition;
  std::vector<size_t> initial_global_assignments;
  size_t slot_start;
  size_t slot_end;
  std::vector<size_t> initial_local_expert_ids;

  size_t sm_version;
  std::vector<size_t> interleave;

  void update() {
    updatePartition();
    updateInterleave();
  }

  void updatePartition() {
    expert_size_per_partition = num_experts / ep_size;
    intermediate_size_per_partition = intermediate_size / tp_size;

    initial_global_assignments.clear();
    for (size_t ep = 0; ep < ep_size; ep++) {
      for (size_t local_slot_id = 0; local_slot_id < expert_size_per_partition; local_slot_id++) {
        initial_global_assignments.push_back((ep_rank * num_experts / ep_size + local_slot_id) % num_experts);
      }
    }
    slot_start = ep_rank * expert_size_per_partition;
    slot_end = slot_start + expert_size_per_partition;
    initial_local_expert_ids = std::vector<size_t>(initial_global_assignments.begin() + slot_start,
                                                   initial_global_assignments.begin() + slot_end);
    KLLM_KERNEL_CHECK(initial_local_expert_ids.size() == expert_size_per_partition);
  }

  void updateInterleave() {
    sm_version = llm_kernels::utils::GetSMVersion();
    if (sm_version == 89) {
      interleave = {1, 1};
    } else if (sm_version == 90) {
      interleave.clear();
      std::vector<size_t> k_shapes = {hidden_size, intermediate_size_per_partition};
      for (const size_t& k_shape : k_shapes) {
        if (k_shape % 512 == 0) {
          interleave.push_back(4);
        } else if (k_shape % 256 == 0) {
          interleave.push_back(2);
        } else if (k_shape % 128 == 0) {
          interleave.push_back(1);
        } else {
          KLLM_KERNEL_THROW(fmt::format("K shape is required to be multiple of 128, received {}.", k_shape));
        }
      }
    } else {
      KLLM_KERNEL_THROW(fmt::format("W4AFP8 MoE is unsupported on SM{}.", sm_version));
    }
  }
};

class NvidiaCutlassMoeTestSuit : public NvidiaTestSuitBase {
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
  const size_t warmup = 100;
  const size_t iters = 1000;

 protected:
  // W4权重专用的unpack方法
  // https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc3/cpp/tensorrt_llm/thop/weightOnlyQuantOp.cpp#L281
  torch::Tensor unpack_int4_packed_tensor_to_int8(torch::Tensor weight) {
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

  // https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc3/tests/unittest/_torch/modules/test_fused_moe.py#L647
  template <typename dtype>
  std::map<std::string, torch::Tensor> createRandomWeight(CutlassFusedMoE model_cfg, float affine_coeff) {
    std::map<std::string, torch::Tensor> weights;

    auto int8_option = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);

    for (size_t expert_id : model_cfg.initial_local_expert_ids) {
      weights[fmt::format("{}.w1.weight", expert_id)] = torch::randint(
          -128, 127,
          {static_cast<int64_t>(model_cfg.intermediate_size), static_cast<int64_t>(model_cfg.hidden_size / 2)},
          int8_option);
      weights[fmt::format("{}.w2.weight", expert_id)] = torch::randint(
          -128, 127,
          {static_cast<int64_t>(model_cfg.hidden_size), static_cast<int64_t>(model_cfg.intermediate_size / 2)},
          int8_option);
      weights[fmt::format("{}.w3.weight", expert_id)] = torch::randint(
          -128, 127,
          {static_cast<int64_t>(model_cfg.intermediate_size), static_cast<int64_t>(model_cfg.hidden_size / 2)},
          int8_option);

      weights[fmt::format("{}.w1.weight_scale_inv", expert_id)] =
          torch::randn({static_cast<int64_t>(model_cfg.intermediate_size),
                        static_cast<int64_t>(model_cfg.hidden_size / model_cfg.scaling_group_size)},
                       dtype_option) *
          affine_coeff;
      weights[fmt::format("{}.w2.weight_scale_inv", expert_id)] =
          torch::randn({static_cast<int64_t>(model_cfg.hidden_size),
                        static_cast<int64_t>(model_cfg.intermediate_size / model_cfg.scaling_group_size)},
                       dtype_option) *
          affine_coeff;
      weights[fmt::format("{}.w3.weight_scale_inv", expert_id)] =
          torch::randn({static_cast<int64_t>(model_cfg.intermediate_size),
                        static_cast<int64_t>(model_cfg.hidden_size / model_cfg.scaling_group_size)},
                       dtype_option) *
          affine_coeff;

      torch::Tensor input_scale =
          torch::randn({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)) * 0.02;
      weights[fmt::format("{}.w1.input_scale", expert_id)] = input_scale.clone();
      weights[fmt::format("{}.w2.input_scale", expert_id)] = input_scale.clone();
      weights[fmt::format("{}.w3.input_scale", expert_id)] = input_scale.clone();
    }

    return weights;
  }

  // https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc3/tensorrt_llm/_torch/modules/fused_moe/routing.py#L95
  std::pair<torch::Tensor, torch::Tensor> renormalizeMoeRoutingMethod(const torch::Tensor& router_logits, int top_k) {
    auto [topk_values, topk_indices] = torch::topk(router_logits, top_k, -1);
    auto indices_int32 = topk_indices.to(torch::kInt32);
    auto softmax_probs = torch::nn::functional::softmax(topk_values.to(torch::kFloat), -1);
    return {indices_int32, softmax_probs};
  }

  template <typename dtype>
  void TestCutlassMoeTacticWithW4AFP8(CutlassFusedMoE model_cfg, size_t tokens_num) {
    using alpha_dtype = float;

    model_cfg.update();

    printf(" ---- Testing %s, tokens_num:%zu, topk:%zu, num_experts:%zu\n", GetDataTypeName<dtype>().c_str(),
           tokens_num, model_cfg.top_k, model_cfg.num_experts);

    auto fused_moe_runner = std::make_shared<FusedMoeRunner>(GetScalarType<dtype>(), ScalarType::QUInt4x2,
                                                             GetScalarType<dtype>(), false, true, false);

    auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);

    BufferMeta w3_w1_weight = CreateBuffer<char>(
        MemoryType::MEMORY_GPU,
        {model_cfg.expert_size_per_partition, model_cfg.intermediate_size_per_partition * 2, model_cfg.hidden_size / 2},
        false);
    BufferMeta w2_weight = CreateBuffer<char>(
        MemoryType::MEMORY_GPU,
        {model_cfg.expert_size_per_partition, model_cfg.hidden_size, model_cfg.intermediate_size_per_partition / 2},
        false);
    BufferMeta fc31_act_scale = CreateBuffer<dtype>(MemoryType::MEMORY_GPU, {1, model_cfg.hidden_size}, false);
    BufferMeta fc2_act_scale =
        CreateBuffer<dtype>(MemoryType::MEMORY_GPU, {1, model_cfg.intermediate_size_per_partition, 1}, false);
    BufferMeta fc31_weight_scale = CreateBuffer<dtype>(
        MemoryType::MEMORY_GPU,
        {model_cfg.expert_size_per_partition, model_cfg.hidden_size / (128 * model_cfg.interleave[0]),
         model_cfg.intermediate_size_per_partition * 2 * model_cfg.interleave[0]},
        false);
    BufferMeta fc2_weight_scale =
        CreateBuffer<dtype>(MemoryType::MEMORY_GPU,
                            {model_cfg.expert_size_per_partition,
                             model_cfg.intermediate_size_per_partition / (128 * model_cfg.interleave[1]),
                             model_cfg.hidden_size * model_cfg.interleave[1]},
                            false);
    BufferMeta fc31_alpha =
        CreateBuffer<alpha_dtype>(MemoryType::MEMORY_GPU, {model_cfg.expert_size_per_partition, 1}, false);
    BufferMeta fc2_alpha =
        CreateBuffer<alpha_dtype>(MemoryType::MEMORY_GPU, {model_cfg.expert_size_per_partition, 1}, false);

    Tensor fc1_expert_weights(w3_w1_weight.data_ptr, w3_w1_weight.shape, ScalarType::QUInt4x2);
    Tensor fc2_expert_weights(w2_weight.data_ptr, w2_weight.shape, ScalarType::QUInt4x2);
    Tensor fc1_weight_scales_tensor(fc31_weight_scale.data_ptr, fc31_weight_scale.shape, GetScalarType<dtype>());
    Tensor fc2_weight_scales_tensor(fc2_weight_scale.data_ptr, fc2_weight_scale.shape, GetScalarType<dtype>());
    Tensor fc1_act_scales_tensor(fc31_act_scale.data_ptr, fc31_act_scale.shape, GetScalarType<dtype>());
    Tensor fc2_act_scales_tensor(fc2_act_scale.data_ptr, fc2_act_scale.shape, GetScalarType<dtype>());
    Tensor fc1_weight_zeros_tensor;
    Tensor fc2_weight_zeros_tensor;
    Tensor fc1_alpha_tensor(fc31_alpha.data_ptr, fc31_alpha.shape, GetScalarType<alpha_dtype>());
    Tensor fc2_alpha_tensor(fc2_alpha.data_ptr, fc2_alpha.shape, GetScalarType<alpha_dtype>());

    int64_t fused_moe_tactic_num = fused_moe_runner->getTacticNum();

    auto get_gemm_best_tactic = [&](int64_t gemm_idx, size_t warmup_iters, size_t profile_iters) -> int64_t {
      // 获取workspace
      size_t profile_workspace_size = fused_moe_runner->getProfileWorkspace(
          fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt, tokens_num, model_cfg.top_k,
          model_cfg.tp_size, model_cfg.tp_rank, model_cfg.ep_size, model_cfg.ep_rank, model_cfg.cluster_size,
          model_cfg.cluster_rank, false, false, gemm_idx, -1, true, stream);
      // 开辟workspace
      BufferMeta profile_workspace = CreateBuffer<char>(MemoryType::MEMORY_GPU, {profile_workspace_size}, false);
      // 设置workspace
      fused_moe_runner->setProfileWorkspace(
          profile_workspace.data_ptr, fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt, tokens_num,
          model_cfg.top_k, model_cfg.tp_size, model_cfg.tp_rank, model_cfg.ep_size, model_cfg.ep_rank,
          model_cfg.cluster_size, model_cfg.cluster_rank, false, false, gemm_idx, -1, true, stream);
      // 获取最优tactic
      int64_t best_tactic = -1;
      float best_tactic_time = std::numeric_limits<float>::max();
      for (int64_t tactic = 0; tactic < fused_moe_tactic_num; tactic++) {
        auto kernel = [&]() {
          fused_moe_runner->runGemmProfile(fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt,
                                           tokens_num, model_cfg.top_k, model_cfg.tp_size, model_cfg.tp_rank,
                                           model_cfg.ep_size, model_cfg.ep_rank, model_cfg.cluster_size,
                                           model_cfg.cluster_rank, false, false, gemm_idx, tactic, false, stream);
        };
        float tactic_time = MeasureCudaExecutionTime(kernel, stream, warmup_iters, profile_iters);
        if (tactic_time < best_tactic_time) {
          best_tactic_time = tactic_time;
          best_tactic = tactic;
        }
      }
      // 释放workspace
      DeleteBuffer(profile_workspace);
      // 结束
      return best_tactic;
    };

    int64_t gemm1_tactic = get_gemm_best_tactic(1, 10, 50);
    int64_t gemm2_tactic = get_gemm_best_tactic(2, 10, 50);

    printf("FusedMoeRunner Tactic Num: %ld, Gemm1 best tactic: %ld, Gemm2 best tactic: %ld\n", fused_moe_tactic_num,
           gemm1_tactic, gemm2_tactic);

    // routing
    torch::Tensor router_logits =
        torch::randn({static_cast<int64_t>(tokens_num), static_cast<int64_t>(model_cfg.num_experts)}, dtype_option);
    auto [topk_ids, topk_weights] = renormalizeMoeRoutingMethod(router_logits, model_cfg.top_k);
    Tensor token_selected_experts_tensor(topk_ids.data_ptr(),
                                         std::vector<size_t>(topk_ids.sizes().begin(), topk_ids.sizes().end()),
                                         GetScalarType<int32_t>());
    Tensor token_final_scales_tensor(topk_weights.data_ptr(),
                                     std::vector<size_t>(topk_weights.sizes().begin(), topk_weights.sizes().end()),
                                     GetScalarType<float>());

    // 推理
    BufferMeta input = CreateBuffer<dtype>(MemoryType::MEMORY_GPU, {tokens_num, model_cfg.hidden_size}, false);
    BufferMeta output = CreateBuffer<dtype>(MemoryType::MEMORY_GPU, {tokens_num, model_cfg.hidden_size}, false);

    Tensor input_tensor(input.data_ptr, input.shape, GetScalarType<dtype>());
    Tensor output_tensor(output.data_ptr, output.shape, GetScalarType<dtype>());

    size_t total_workspace_size = fused_moe_runner->getRuntimeWorkspaceInfo(
        input_tensor, token_selected_experts_tensor, fc2_expert_weights, model_cfg.tp_size, 0, model_cfg.ep_size, 0,
        false, {gemm1_tactic, gemm2_tactic});
    BufferMeta workspace = CreateBuffer<char>(MemoryType::MEMORY_GPU, {total_workspace_size}, false);
    fused_moe_runner->setRuntimeWorkspaceInfo(workspace.data_ptr);
    printf("Runtime Workspace Size: %zu\n", total_workspace_size);

    auto kernel = [&]() {
      fused_moe_runner->runMoe(
          output_tensor, input_tensor, token_selected_experts_tensor, token_final_scales_tensor, fc1_expert_weights,
          std::nullopt, fc2_expert_weights, std::nullopt,
          {fc1_weight_scales_tensor, fc2_weight_scales_tensor, fc1_act_scales_tensor, fc2_act_scales_tensor,
           fc1_weight_zeros_tensor, fc2_weight_zeros_tensor, fc1_alpha_tensor, fc2_alpha_tensor},
          std::nullopt, model_cfg.tp_size, 0, model_cfg.ep_size, 0, 1, 0, false, false, {gemm1_tactic, gemm2_tactic},
          stream);
    };
    float kernel_time = MeasureCudaExecutionTime(kernel, stream, 100, 1000);

    printf("Kernel time: %f ms\n", kernel_time);
  }

  template <typename dtype>
  void TestCutlassMoePrecisionWithW4AFP8(CutlassFusedMoE model_cfg, size_t tokens_num) {
    using alpha_dtype = float;

    model_cfg.update();

    printf(" ---- Testing %s, tokens_num:%zu, topk:%zu, num_experts:%zu\n", GetDataTypeName<dtype>().c_str(),
           tokens_num, model_cfg.top_k, model_cfg.num_experts);

    auto int8_option = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);
    auto alpha_dtype_option = torch::TensorOptions().dtype(GetTorchDataType<alpha_dtype>()).device(torch::kCUDA);
    auto float_option = torch::TensorOptions().dtype(GetTorchDataType<float>()).device(torch::kCUDA);

    // 创建输入数据
    torch::Tensor x =
        torch::randn({static_cast<int64_t>(tokens_num), static_cast<int64_t>(model_cfg.hidden_size)}, dtype_option);
    torch::Tensor router_logits =
        torch::randn({static_cast<int64_t>(tokens_num), static_cast<int64_t>(model_cfg.num_experts)}, dtype_option);
    torch::Tensor e_bias = torch::ones({static_cast<int64_t>(model_cfg.num_experts)}, float_option);
    torch::Tensor cutlass_output = torch::zeros_like(x);

    // routing
    auto [topk_ids, topk_weights] = renormalizeMoeRoutingMethod(router_logits, model_cfg.top_k);

    // 创建权重
    std::map<std::string, torch::Tensor> weights = createRandomWeight<dtype>(model_cfg, 0.005);

    // 模拟权重加载
    // NOTE(jinxcwu): sm90专用，sm89会有一些额外的操作
    // https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc3/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L549
    // 步骤1: 创建一份空权重
    torch::Tensor w3_w1_weight = torch::empty({static_cast<int64_t>(model_cfg.expert_size_per_partition),
                                               static_cast<int64_t>(model_cfg.intermediate_size_per_partition * 2),
                                               static_cast<int64_t>(model_cfg.hidden_size / 2)},
                                              int8_option);
    torch::Tensor w2_weight = torch::empty(
        {static_cast<int64_t>(model_cfg.expert_size_per_partition), static_cast<int64_t>(model_cfg.hidden_size),
         static_cast<int64_t>(model_cfg.intermediate_size_per_partition / 2)},
        int8_option);
    torch::Tensor fc31_act_scale =
        torch::empty({static_cast<int64_t>(1), static_cast<int64_t>(model_cfg.hidden_size)}, dtype_option);
    torch::Tensor fc2_act_scale =
        torch::empty({static_cast<int64_t>(1), static_cast<int64_t>(model_cfg.intermediate_size_per_partition),
                      static_cast<int64_t>(1)},
                     dtype_option);
    torch::Tensor fc31_weight_scale =
        torch::empty({static_cast<int64_t>(model_cfg.expert_size_per_partition),
                      static_cast<int64_t>(model_cfg.hidden_size / (128 * model_cfg.interleave[0])),
                      static_cast<int64_t>(model_cfg.intermediate_size_per_partition * 2 * model_cfg.interleave[0])},
                     dtype_option);
    torch::Tensor fc2_weight_scale =
        torch::empty({static_cast<int64_t>(model_cfg.expert_size_per_partition),
                      static_cast<int64_t>(model_cfg.intermediate_size_per_partition / (128 * model_cfg.interleave[1])),
                      static_cast<int64_t>(model_cfg.hidden_size * model_cfg.interleave[1])},
                     dtype_option);
    torch::Tensor fc31_alpha = torch::empty(
        {static_cast<int64_t>(model_cfg.expert_size_per_partition), static_cast<int64_t>(1)}, alpha_dtype_option);
    torch::Tensor fc2_alpha = torch::empty(
        {static_cast<int64_t>(model_cfg.expert_size_per_partition), static_cast<int64_t>(1)}, alpha_dtype_option);
    // 步骤2 处理weight
    for (size_t local_slot_id = 0; local_slot_id < model_cfg.initial_local_expert_ids.size(); local_slot_id++) {
      size_t expert_id = model_cfg.initial_local_expert_ids[local_slot_id];
      w3_w1_weight[local_slot_id].copy_(torch::cat(
          {weights[fmt::format("{}.w3.weight", expert_id)], weights[fmt::format("{}.w1.weight", expert_id)]}, 0));
      w2_weight[local_slot_id].copy_(weights[fmt::format("{}.w2.weight", expert_id)]);
    }
    // 步骤3 处理act scale和alpha
    float all_w2_input_scales_max = std::numeric_limits<float>::min();
    for (size_t local_slot_id = 0; local_slot_id < model_cfg.initial_local_expert_ids.size(); local_slot_id++) {
      size_t expert_id = model_cfg.initial_local_expert_ids[local_slot_id];
      all_w2_input_scales_max = std::max(
          all_w2_input_scales_max, torch::max(weights[fmt::format("{}.w2.input_scale", expert_id)]).item<float>());
    }
    fc2_act_scale.copy_((torch::ones_like(fc2_act_scale) * (1 / all_w2_input_scales_max)).to(fc2_act_scale.dtype()));
    fc2_alpha.copy_((torch::ones_like(fc2_alpha) * all_w2_input_scales_max).to(fc2_alpha.dtype()));
    // 步骤4 处理act scale和alpha
    float all_w3_w1_input_scales_max = std::numeric_limits<float>::min();
    for (size_t local_slot_id = 0; local_slot_id < model_cfg.initial_local_expert_ids.size(); local_slot_id++) {
      size_t expert_id = model_cfg.initial_local_expert_ids[local_slot_id];
      all_w3_w1_input_scales_max = std::max(
          all_w3_w1_input_scales_max, torch::max(weights[fmt::format("{}.w3.input_scale", expert_id)]).item<float>());
      all_w3_w1_input_scales_max = std::max(
          all_w3_w1_input_scales_max, torch::max(weights[fmt::format("{}.w1.input_scale", expert_id)]).item<float>());
    }
    fc31_act_scale.copy_(
        (torch::ones_like(fc31_act_scale) * (1 / all_w3_w1_input_scales_max)).to(fc31_act_scale.dtype()));
    fc31_alpha.copy_((torch::ones_like(fc31_alpha) * all_w3_w1_input_scales_max).to(fc31_alpha.dtype()));
    // 步骤5
    std::vector<torch::Tensor> all_w2_scales;
    for (size_t local_slot_id = 0; local_slot_id < model_cfg.initial_local_expert_ids.size(); local_slot_id++) {
      size_t expert_id = model_cfg.initial_local_expert_ids[local_slot_id];
      all_w2_scales.push_back(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    torch::Tensor w2_scales = torch::stack(all_w2_scales).to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w2_s_shape = w2_scales.sizes();
    torch::Tensor w2_scales_interleaved =
        w2_scales.reshape({w2_s_shape[0], w2_s_shape[1], w2_s_shape[2] / static_cast<int64_t>(model_cfg.interleave[1]),
                           static_cast<int64_t>(model_cfg.interleave[1])});
    w2_scales_interleaved = w2_scales_interleaved.permute({0, 2, 1, 3});
    w2_scales_interleaved =
        w2_scales_interleaved.reshape({w2_s_shape[0], w2_s_shape[2] / static_cast<int64_t>(model_cfg.interleave[1]),
                                       w2_s_shape[1] * static_cast<int64_t>(model_cfg.interleave[1])});
    fc2_weight_scale.copy_(w2_scales_interleaved.contiguous());
    // 步骤6
    std::vector<torch::Tensor> all_w3_scales;
    std::vector<torch::Tensor> all_w1_scales;
    for (size_t local_slot_id = 0; local_slot_id < model_cfg.initial_local_expert_ids.size(); local_slot_id++) {
      size_t expert_id = model_cfg.initial_local_expert_ids[local_slot_id];
      all_w3_scales.push_back(weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]);
      all_w1_scales.push_back(weights[fmt::format("{}.w1.weight_scale_inv", expert_id)]);
    }
    torch::Tensor all_w3_scales_stack = torch::stack(all_w3_scales);
    torch::Tensor all_w1_scales_stack = torch::stack(all_w1_scales);
    torch::Tensor all_w3_w1_scales = torch::cat({all_w3_scales_stack, all_w1_scales_stack}, -2);
    torch::Tensor w3_w1_scales = all_w3_w1_scales.to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w3_w1_s_shape = w3_w1_scales.sizes();
    torch::Tensor w3_w1_scales_interleaved = w3_w1_scales.reshape(
        {w3_w1_s_shape[0], w3_w1_s_shape[1], w3_w1_s_shape[2] / static_cast<int64_t>(model_cfg.interleave[0]),
         static_cast<int64_t>(model_cfg.interleave[0])});
    w3_w1_scales_interleaved = w3_w1_scales_interleaved.permute({0, 2, 1, 3});
    w3_w1_scales_interleaved = w3_w1_scales_interleaved.reshape(
        {w3_w1_s_shape[0], w3_w1_s_shape[2] / static_cast<int64_t>(model_cfg.interleave[0]),
         w3_w1_s_shape[1] * static_cast<int64_t>(model_cfg.interleave[0])});
    fc31_weight_scale.copy_(w3_w1_scales_interleaved.contiguous());

    // Tensor转换
    Tensor input_tensor(x.data_ptr(), std::vector<size_t>(x.sizes().begin(), x.sizes().end()), GetScalarType<dtype>());
    Tensor cutlass_output_tensor(cutlass_output.data_ptr(),
                                 std::vector<size_t>(cutlass_output.sizes().begin(), cutlass_output.sizes().end()),
                                 GetScalarType<dtype>());
    Tensor token_selected_experts_tensor(topk_ids.data_ptr(),
                                         std::vector<size_t>(topk_ids.sizes().begin(), topk_ids.sizes().end()),
                                         GetScalarType<int32_t>());
    Tensor token_final_scales_tensor(topk_weights.data_ptr(),
                                     std::vector<size_t>(topk_weights.sizes().begin(), topk_weights.sizes().end()),
                                     GetScalarType<float>());
    Tensor fc1_expert_weights_tensor(w3_w1_weight.data_ptr(),
                                     std::vector<size_t>(w3_w1_weight.sizes().begin(), w3_w1_weight.sizes().end()),
                                     ScalarType::QUInt4x2);
    Tensor fc2_expert_weights_tensor(w2_weight.data_ptr(),
                                     std::vector<size_t>(w2_weight.sizes().begin(), w2_weight.sizes().end()),
                                     ScalarType::QUInt4x2);
    Tensor fc1_weight_scales_tensor(
        fc31_weight_scale.data_ptr(),
        std::vector<size_t>(fc31_weight_scale.sizes().begin(), fc31_weight_scale.sizes().end()),
        GetScalarType<dtype>());
    Tensor fc2_weight_scales_tensor(
        fc2_weight_scale.data_ptr(),
        std::vector<size_t>(fc2_weight_scale.sizes().begin(), fc2_weight_scale.sizes().end()), GetScalarType<dtype>());
    Tensor fc1_act_scales_tensor(fc31_act_scale.data_ptr(),
                                 std::vector<size_t>(fc31_act_scale.sizes().begin(), fc31_act_scale.sizes().end()),
                                 GetScalarType<dtype>());
    Tensor fc2_act_scales_tensor(fc2_act_scale.data_ptr(),
                                 std::vector<size_t>(fc2_act_scale.sizes().begin(), fc2_act_scale.sizes().end()),
                                 GetScalarType<dtype>());
    Tensor fc1_weight_zeros_tensor;
    Tensor fc2_weight_zeros_tensor;
    Tensor fc1_alpha_tensor(fc31_alpha.data_ptr(),
                            std::vector<size_t>(fc31_alpha.sizes().begin(), fc31_alpha.sizes().end()),
                            GetScalarType<alpha_dtype>());
    Tensor fc2_alpha_tensor(fc2_alpha.data_ptr(),
                            std::vector<size_t>(fc2_alpha.sizes().begin(), fc2_alpha.sizes().end()),
                            GetScalarType<alpha_dtype>());

    // 创建cutlass moe算子
    auto fused_moe_runner = std::make_shared<FusedMoeRunner>(GetScalarType<dtype>(), ScalarType::QUInt4x2,
                                                             GetScalarType<dtype>(), false, true, false);
    // 创建workspace
    size_t total_workspace_size = fused_moe_runner->getRuntimeWorkspaceInfo(
        input_tensor, token_selected_experts_tensor, fc2_expert_weights_tensor, model_cfg.tp_size, model_cfg.tp_rank,
        model_cfg.ep_size, model_cfg.ep_rank, false, {});
    BufferMeta workspace = CreateBuffer<char>(MemoryType::MEMORY_GPU, {total_workspace_size}, false);
    fused_moe_runner->setRuntimeWorkspaceInfo(workspace.data_ptr);
    // cutlass推理获取结果
    fused_moe_runner->runMoe(
        cutlass_output_tensor, input_tensor, token_selected_experts_tensor, token_final_scales_tensor,
        fc1_expert_weights_tensor, std::nullopt, fc2_expert_weights_tensor, std::nullopt,
        {fc1_weight_scales_tensor, fc2_weight_scales_tensor, fc1_act_scales_tensor, fc2_act_scales_tensor,
         fc1_weight_zeros_tensor, fc2_weight_zeros_tensor, fc1_alpha_tensor, fc2_alpha_tensor},
        std::nullopt, model_cfg.tp_size, model_cfg.tp_rank, model_cfg.ep_size, model_cfg.ep_rank,
        model_cfg.cluster_size, model_cfg.cluster_rank, false, false, {}, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 计算参考结果
    torch::Tensor selected_experts = topk_ids.clone();
    torch::Tensor final_scales = topk_weights.clone();
    torch::Tensor ref_output = torch::zeros_like(x);
    for (size_t e_idx = 0; e_idx < model_cfg.num_experts; e_idx++) {
      torch::Tensor mask = selected_experts == e_idx;
      torch::Tensor activated_tokens = mask.sum(1).to(torch::kBool);
      torch::Tensor act = x.index_select(0, activated_tokens.nonzero().squeeze());
      if (act.size(0) == 0) {
        continue;
      }
      torch::Tensor final_scale = (final_scales * mask.to(torch::kFloat))
                                      .sum(1)
                                      .index_select(0, activated_tokens.nonzero().squeeze())
                                      .unsqueeze(1);

      torch::Tensor w1 = weights[fmt::format("{}.w1.weight", e_idx)];
      w1 = unpack_int4_packed_tensor_to_int8(w1.cpu()).t().contiguous().cuda();
      torch::Tensor w2 = weights[fmt::format("{}.w2.weight", e_idx)];
      w2 = unpack_int4_packed_tensor_to_int8(w2.cpu()).t().contiguous().cuda();
      torch::Tensor w3 = weights[fmt::format("{}.w3.weight", e_idx)];
      w3 = unpack_int4_packed_tensor_to_int8(w3.cpu()).t().contiguous().cuda();
      torch::Tensor w3_w1 = torch::cat({w3, w1}, -1);

      torch::Tensor s1 = weights[fmt::format("{}.w1.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s2 = weights[fmt::format("{}.w2.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s3 = weights[fmt::format("{}.w3.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s3_s1 = torch::cat({s3, s1}, -1);

      torch::Tensor p1 = weights[fmt::format("{}.w1.input_scale", e_idx)];
      torch::Tensor p2 = weights[fmt::format("{}.w2.input_scale", e_idx)];
      torch::Tensor p3 = weights[fmt::format("{}.w3.input_scale", e_idx)];
      torch::Tensor p3_p1 = torch::max(p1, p3);

      act = torch::clamp((act / p3_p1), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w3_w1 =
          (w3_w1.to(torch::kFloat) * s3_s1.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc1_gate = torch::matmul(act, w3_w1) * p3_p1;
      auto chunks = fc1_gate.chunk(2, -1);
      torch::Tensor fc1 = chunks[0];
      torch::Tensor gate = chunks[1];
      fc1 = fc1 * torch::nn::functional::silu(gate);

      act = torch::clamp((fc1 / p2), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w2 = (w2.to(torch::kFloat) * s2.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc2 = torch::matmul(act, w2) * p2;
      ref_output.index_add_(0, activated_tokens.nonzero().squeeze(), (fc2 * final_scale).to(ref_output.dtype()));
    }

    // 结果检查
    BufferMeta c_o = CreateBuffer<dtype>(MemoryType::MEMORY_GPU, cutlass_output_tensor.shape, true);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(c_o.data_ptr, cutlass_output.data_ptr(),
                                       tokens_num * model_cfg.hidden_size * sizeof(dtype), cudaMemcpyDeviceToDevice));
    BufferMeta r_o = CreateBuffer<dtype>(MemoryType::MEMORY_GPU, cutlass_output_tensor.shape, true);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(r_o.data_ptr, ref_output.data_ptr(),
                                       tokens_num * model_cfg.hidden_size * sizeof(dtype), cudaMemcpyDeviceToDevice));
    EXPECT_TRUE(CheckResult<dtype>("CheckResult", c_o, r_o, 0.1f, 1e-2f));
  }
};

TEST_F(NvidiaCutlassMoeTestSuit, TestCutlassMoeTacticWithW4AFP8) {
  CutlassFusedMoE r1_cfg;
  r1_cfg.ep_size = 1;
  r1_cfg.ep_rank = 0;
  r1_cfg.tp_size = 1;
  r1_cfg.tp_rank = 0;
  r1_cfg.scaling_group_size = 128;
  r1_cfg.top_k = 8;
  r1_cfg.num_experts = 32;
  r1_cfg.hidden_size = 768;
  r1_cfg.intermediate_size = 640;

  std::vector<size_t> tokens_num = {1, 4, 16, 32, 64};

  for (auto token_num : tokens_num) {
    TestCutlassMoeTacticWithW4AFP8<__nv_bfloat16>(r1_cfg, token_num);
    TestCutlassMoeTacticWithW4AFP8<half>(r1_cfg, token_num);
  }
}

TEST_F(NvidiaCutlassMoeTestSuit, TestCutlassMoePrecisionWithW4AFP8) {
  CutlassFusedMoE r1_cfg;
  r1_cfg.ep_size = 1;
  r1_cfg.ep_rank = 0;
  r1_cfg.tp_size = 1;
  r1_cfg.tp_rank = 0;
  r1_cfg.scaling_group_size = 128;
  r1_cfg.top_k = 8;
  r1_cfg.num_experts = 32;
  r1_cfg.hidden_size = 768;
  r1_cfg.intermediate_size = 640;

  std::vector<size_t> tokens_num = {1, 4, 16, 32, 64};

  for (auto token_num : tokens_num) {
    TestCutlassMoePrecisionWithW4AFP8<__nv_bfloat16>(r1_cfg, token_num);
    TestCutlassMoePrecisionWithW4AFP8<half>(r1_cfg, token_num);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels