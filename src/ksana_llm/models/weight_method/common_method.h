/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "ksana_llm/models/base/common_model_weight_loader.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class CommonMethod {
 public:
  CommonMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp)
      : common_weight_loader_(common_weight_loader), tp_(tp) {}

  Status load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                          const Tensor& weight_tensor, int dev_rank);

  Status load_attn_norm(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                        const Tensor& weight_tensor, int dev_rank);

  Status load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_norm(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                   const Tensor& weight_tensor, int dev_rank);

  Status load_embed_tokens(std::unordered_map<std::string, Tensor>& device_model_weights,
                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status load_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
                      const Tensor& weight_tensor, int dev_rank);

  Status process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  Status process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_prefix_name, int dev_rank);

  Status process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_prefix_name, int dev_rank);

  Status process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                             const std::string& weight_prefix_name, int dev_rank);

  Status process_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights,
                         const std::string& weight_prefix_name, int dev_rank);

 private:
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  int tp_;
};

}  // namespace ksana_llm