/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once
#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/moe/cutlass_moe/cutlass_moe_wrapper.h"
#  include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#endif
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class CutlassMoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  inline Status ProcessChunks(const Tensor& input_tensor, Tensor& output_tensor, const size_t total_tokens,
                              const llm_kernels::nvidia::tensorrt_llm::dev::Tensor& fc1_expert_weights_ktensor,
                              const llm_kernels::nvidia::tensorrt_llm::dev::Tensor& fc2_expert_weights_ktensor,
                              const std::vector<llm_kernels::nvidia::tensorrt_llm::dev::Tensor>& quant_scales_ktensors);

  // 执行 GroupedTopk 计算的辅助函数
  Status ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Used for distributing and gathering topk_ids and hidden_buffer in Expert-Parallel scenarios
  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // Used for Expert-Parallel load balancer.
  Status DumpEplbData(Tensor& topk_ids);

  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

 protected:
  bool set_workspace_buffer_info_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
  size_t max_token_num_;
  size_t expert_num_per_node_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  int layer_idx_;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;
  bool enable_full_shared_expert_ = false;
  int group_size_;
  bool apply_weight_ = false;

  void* topk_weights_ptr_;
  size_t topk_weights_ptr_size;
  void* topk_ids_ptr_;
  size_t topk_ids_ptr_size;
  void* kernel_workspace_ptr_;
  size_t kernel_workspace_size;

  size_t global_expert_para_size_;
  size_t global_expert_para_rank_;

  bool using_deepep_;

  std::shared_ptr<llm_kernels::nvidia::moe::ExpertMap> expert_map_;

  // Used for Expert-Parallel load balancer.
  bool enable_load_eplb_weight_ = false;
  bool enable_dump_eplb_data_ = false;
  int eplb_dump_step_ = 0;
  std::string eplb_dump_path_;

  std::vector<std::vector<int64_t>> config_map_;

  std::shared_ptr<GroupedTopkLayer> grouped_topk_layer_;

  std::shared_ptr<llm_kernels::nvidia::CutlassMoeWrapper> cutlass_moe_wrapper_;
};
#endif
}  // namespace ksana_llm
