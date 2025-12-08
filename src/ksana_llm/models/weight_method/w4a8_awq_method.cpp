/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/weight_method/w4a8_awq_method.h"

#include <torch/torch.h>

#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/models/base/utils.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/gemm/finegrained_mixed_dtype_gemm/finegrained_mixed_dtype_gemm_wrapper.h"
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#endif

namespace ksana_llm {

Status W4A8AWQMethod::load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".input_scale", ".weight_scale_2", ".pre_quant_scale"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  if (CheckWeightNameEndMatched(weight_name, {".weight", ".weight_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status W4A8AWQMethod::load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".input_scale", ".weight_scale_2"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  if (CheckWeightNameEndMatched(weight_name, {".weight", ".weight_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  if (CheckWeightNameEndMatched(weight_name, {".pre_quant_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status W4A8AWQMethod::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".input_scale", ".weight_scale_2", ".pre_quant_scale"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  if (CheckWeightNameEndMatched(weight_name, {".weight", ".weight_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status W4A8AWQMethod::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".input_scale", ".weight_scale_2"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  if (CheckWeightNameEndMatched(weight_name, {".weight", ".weight_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  if (CheckWeightNameEndMatched(weight_name, {".pre_quant_scale"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status W4A8AWQMethod::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_proj_name = weight_prefix_name;
  const std::string k_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "k_proj");
  const std::string v_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "v_proj");
  const std::string qkv_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "query_key_value");
  // 合并qkv
  {
    const std::vector<std::string> weight_suffixs = {"weight", "weight_scale"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight(
          {q_proj_name + weight_suffix, k_proj_name + weight_suffix, v_proj_name + weight_suffix},
          qkv_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  // 获取共享权重
  {
    const std::vector<std::string> weight_suffixs = {"input_scale", "weight_scale_2"};
    for (const std::string& weight_suffix : weight_suffixs) {
      const std::string q_name = q_proj_name + weight_suffix;
      const std::string k_name = k_proj_name + weight_suffix;
      const std::string v_name = v_proj_name + weight_suffix;
      const std::string qkv_name = qkv_proj_name + weight_suffix;
      device_model_weights[qkv_name] = device_model_weights[q_name];
      device_model_weights.erase(q_name);
      device_model_weights.erase(k_name);
      device_model_weights.erase(v_name);
    }
  }
  // 权重转换与绑定
  convert_weight_and_bind(device_model_weights, qkv_proj_name, dev_rank);
  return Status();
}

Status W4A8AWQMethod::process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                               const std::string& weight_prefix_name, int dev_rank) {
  const std::string gate_proj_name = weight_prefix_name;
  const std::string up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "up_proj");
  const std::string gate_up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "gate_up_proj");
  // 合并gate_up
  {
    const std::vector<std::string> weight_suffixs = {"weight", "weight_scale"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight({gate_proj_name + weight_suffix, up_proj_name + weight_suffix},
                                             gate_up_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  // 获取共享权重
  {
    const std::vector<std::string> weight_suffixs = {"input_scale", "weight_scale_2"};
    for (const std::string& weight_suffix : weight_suffixs) {
      const std::string gate_name = gate_proj_name + weight_suffix;
      const std::string up_name = up_proj_name + weight_suffix;
      const std::string gate_up_name = gate_up_proj_name + weight_suffix;
      device_model_weights[gate_up_name] = device_model_weights[gate_name];
      device_model_weights.erase(gate_name);
      device_model_weights.erase(up_name);
    }
  }
  // 权重转换与绑定
  convert_weight_and_bind(device_model_weights, gate_up_proj_name, dev_rank);
  return Status();
}

Status W4A8AWQMethod::process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_prefix_name, int dev_rank) {
  // 权重转换与绑定
  convert_weight_and_bind(device_model_weights, weight_prefix_name, dev_rank);
  return Status();
}

Status W4A8AWQMethod::process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_prefix_name, int dev_rank) {
  // 权重转换与绑定
  convert_weight_and_bind(device_model_weights, weight_prefix_name, dev_rank);
  return Status();
}

Status W4A8AWQMethod::convert_weight_and_bind(std::unordered_map<std::string, Tensor>& device_model_weights,
                                              const std::string& weight_prefix_name, int dev_rank) {
#ifdef ENABLE_CUDA
  // 读取并转换为torch::Tensor
  const std::string weight_name = weight_prefix_name + "weight";
  const std::string weight_scale_name = weight_prefix_name + "weight_scale";
  const std::string input_scale_name = weight_prefix_name + "input_scale";
  const std::string weight_scale_2_name = weight_prefix_name + "weight_scale_2";
  const std::string pre_quant_scale_name = weight_prefix_name + "pre_quant_scale";
  // 转置
  common_weight_loader_->Permute2D(device_model_weights.at(weight_scale_name), dev_rank);
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  // 获取torch::Tensor
  torch::Tensor input_scale = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[input_scale_name]);
  torch::Tensor weight_scale_2 =
      common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[weight_scale_2_name]);
  torch::Tensor weight_scale = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[weight_scale_name]);
  torch::Tensor weight = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[weight_name]);
  // 计算并获取alpha
  float alpha = (input_scale.to(torch::kFloat32) * weight_scale_2.to(torch::kFloat32)).item<float>();
  // weight scale 转换、更新
  // ref: https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc4/tensorrt_llm/_torch/modules/linear.py#L1615
  weight_scale = weight_scale / weight_scale_2;
  device_model_weights[weight_scale_name] = common_weight_loader_->GetTensorFromTorchTensor(weight_scale);
  common_weight_loader_->CastDeviceTensorType(device_model_weights[weight_scale_name], DataType::TYPE_FP16, dev_rank);
  // weight 转换、更新
  // TODO(jinxcwu) PreprocessWeightsForMixedGemm的使用可以再优化一下
  using KScalarType = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType;
  llm_kernels::nvidia::FinegrainedMixedDtypeGemmWrapper wrapper(KScalarType::Float8_e4m3fn, KScalarType::BFloat16,
                                                                llm_kernels::nvidia::QuantMode::NO_ZERO, 128);
  torch::Tensor processed_weight = wrapper.PreprocessWeightsForMixedGemm(weight.view(torch::kInt8), torch::kQUInt4x2,
                                                                         torch::kFloat8_e4m3fn, -1, true);
  device_model_weights[weight_name] = common_weight_loader_->GetTensorFromTorchTensor(processed_weight);
  // 删除weight_scale_2
  device_model_weights.erase(weight_scale_2_name);
  // 绑定
  device_model_weights[weight_name].input_scales = &device_model_weights[input_scale_name];
  device_model_weights[weight_name].weight_scales = &device_model_weights[weight_scale_name];
  device_model_weights[weight_name].alpha = alpha;
  // pre_quant_scale的特殊处理
  if (device_model_weights.find(pre_quant_scale_name) != device_model_weights.end()) {
    torch::Tensor pre_quant_scale =
        common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[pre_quant_scale_name]);
    // 融合pre_quant_scale与input_scale
    pre_quant_scale = input_scale.to(torch::kFloat32) / pre_quant_scale.to(torch::kFloat32);
    device_model_weights[pre_quant_scale_name] = common_weight_loader_->GetTensorFromTorchTensor(pre_quant_scale);
    device_model_weights[weight_name].pre_quant_scales = &device_model_weights[pre_quant_scale_name];
  }
  // 设置weight status到context
  WeightStatus weight_status;
  weight_status.quant_mode = QuantMode::QUANT_W4A8_AWQ;
  weight_status.layout["k"] = static_cast<size_t>(processed_weight.size(0));
  weight_status.layout["n"] = static_cast<size_t>(2 * processed_weight.size(1));
  common_weight_loader_->GetContext()->SetWeightStatus(weight_name, weight_status);
  return Status();
#else
  return Status(RET_RUNTIME_FAILED, "W4A8 AWQ quantization is only supported on CUDA devices");
#endif
}

}  // namespace ksana_llm