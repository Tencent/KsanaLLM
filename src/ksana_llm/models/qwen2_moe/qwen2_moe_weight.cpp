/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/qwen2_moe/qwen2_moe_weight.h"

namespace ksana_llm {

template <typename T>
Qwen2MoeWeight<T>::Qwen2MoeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                                  std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context),
      CommonMoeWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status Qwen2MoeWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                              const std::vector<std::string>& weight_name_list,
                                              const std::vector<std::string>& custom_name_list) {
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  return Status();
}

template <typename T>
Status Qwen2MoeWeight<T>::PermuteShareGatingWeight(Tensor& last_share_gating_tensor, const int num_layer) {
  SetDevice(rank_);
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string gating_name = "model.layers." + std::to_string(layer_idx) + ".mlp.shared_expert_gate.weight";
    CommonWeight<T>::CommonPermuteWeight(gating_name, last_share_gating_tensor);
  }
  return Status();
}

template <typename T>
void Qwen2MoeWeight<T>::ProcessWeights() {
  CommonMoeWeight<T>::ProcessWeights();
  int num_layers = model_config_.num_layer;
  // Permute share gating Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.shared_expert_gate.weight",
                                             "empty_share_gating_tensor");
  Tensor& last_share_gating_tensor = weights_map_["empty_share_gating_tensor"];
  PermuteShareGatingWeight(last_share_gating_tensor, num_layers);
  weights_map_.erase("empty_share_gating_tensor");

  CommonWeight<T>::ProcessWeights();
}

template class Qwen2MoeWeight<float>;
template class Qwen2MoeWeight<float16>;
template class Qwen2MoeWeight<bfloat16>;

}  // namespace ksana_llm
