/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

struct WorkspaceInfo {
  void* workspace{};
  void* scale_probs{};
  void* fc2_output{};
  void* src_to_dest_map{};
  void* selected_experts{};
  void* lora_workspace{};
  size_t size{};
};
#ifdef ENABLE_CUDA
template <typename T>
class MoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  bool set_workspace_buffer_info_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
  size_t max_token_num_;
  size_t expert_num_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  bool use_lora_ = false;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;

  size_t scale_probabilities_size_;
  size_t src_to_dest_map_size_;
  size_t selected_expert_size_;
  size_t lora_workspace_size_;
  size_t moe_workspace_size_;

  bool apply_weight_ = false;

  void* topk_weights_ptr_;
  size_t topk_weights_ptr_size;
  void* topk_ids_ptr_;
  size_t topk_ids_ptr_size;
  void* intermediate_cache1_;
  size_t intermediate_cache1_size;
  void* intermediate_cache2_;
  size_t intermediate_cache2_size;
  void* intermediate_cache3_;
  size_t intermediate_cache3_size;

  // buffer_of_ids_in_kernel
  void* fused_id_buffer_;
  size_t max_fused_id_buffer_size;

  // shared space size
  size_t intermediate_cache1_and_cache3_size;

  DataType weight_dtype_;
  DataType compute_dtype_;
  std::vector<int> block_shape_;
  void* a1_q_;
  void* a2_q_;
  void* a1_scale_;
  void* a2_scale_;
  size_t a1_q_size;
  size_t a2_q_size;
  size_t a1_scale_size;
  size_t a2_scale_size;
  size_t a1_and_a2_q_size;
  size_t a1_and_a2_scale_size;

  size_t dequant_workspace_size;
  void* dequant_workspace_;

  // The vector of the best config index for every tokens number
  std::vector<size_t> config_map_;
  WorkspaceInfo workspace_info_;
};
#endif
}  // namespace ksana_llm
