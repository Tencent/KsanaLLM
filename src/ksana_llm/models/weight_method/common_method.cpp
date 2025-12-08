/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/weight_method/common_method.h"

#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/models/base/utils.h"

namespace ksana_llm {

Status CommonMethod::load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status CommonMethod::load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_attn_norm(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status CommonMethod::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status CommonMethod::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                        const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_norm(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status CommonMethod::load_embed_tokens(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status CommonMethod::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_proj_name = weight_prefix_name;
  const std::string k_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "k_proj");
  const std::string v_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "v_proj");
  const std::string qkv_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "query_key_value");
  // 合并qkv
  {
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight(
          {q_proj_name + weight_suffix, k_proj_name + weight_suffix, v_proj_name + weight_suffix},
          qkv_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  const std::string weight_name = qkv_proj_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status CommonMethod::process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                              const std::string& weight_prefix_name, int dev_rank) {
  const std::string gate_proj_name = weight_prefix_name;
  const std::string up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "up_proj");
  const std::string gate_up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "gate_up_proj");
  // 合并gate_up
  {
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight({gate_proj_name + weight_suffix, up_proj_name + weight_suffix},
                                             gate_up_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  const std::string weight_name = gate_up_proj_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status CommonMethod::process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status CommonMethod::process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status CommonMethod::process_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights,
                                     const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

}  // namespace ksana_llm