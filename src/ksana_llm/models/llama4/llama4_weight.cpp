/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/llama4/llama4_weight.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/kernels/trans_layout.h"

namespace ksana_llm {

template <typename T>
Llama4Weight<T>::Llama4Weight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context), CommonMoeWeight<T>(model_config, rank, context) {}

template <typename T>
Status Llama4Weight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                            const std::vector<std::string>& weight_name_list,
                                            const std::vector<std::string>& custom_name_list) {
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  return Status();
}

template <typename T>
Status Llama4Weight<T>::PermuteExpertsWeight() {
  SetDevice(rank_);
  // permute(0,2,1) for experts.up_gate_proj.weight
  // src[num_experts, hidden_units, 2 * moe_inter_size_per_rank]
  // dst[num_experts, 2 * moe_inter_size_per_rank, hidden_units]
  // permute(0,2,1) for experts.down_proj.weight
  // src[num_experts, moe_inter_size_per_rank, hidden_units]
  // dst[num_experts, hidden_units, moe_inter_size_per_rank]
  std::vector<std::string> names = {"experts.up_gate_proj", "experts.down_proj"};
  int num_layers = model_config_.num_layer;
  for (std::string& name : names) {
    std::string swap_tensor_name = "empty_" + name + "_tensor";
    tensor_manager_->CreateTensorWithSameShape(
        "model.layers." + std::to_string(*required_layer_idx_.moe.begin()) + ".mlp." + name + ".weight",
        swap_tensor_name);
    Tensor& swap_tensor = weights_map_[swap_tensor_name];
    for (const auto layer_idx : required_layer_idx_.moe) {
      std::string origin_tensor_name = "model.layers." + std::to_string(layer_idx) + ".mlp." + name + ".weight";
      Tensor& origin_tensor = weights_map_[origin_tensor_name];
      Permute(origin_tensor, swap_tensor, {0, 2, 1}, context_->GetMemoryManageStreams()[rank_]);
      Tensor t = swap_tensor;
      swap_tensor = origin_tensor;
      t.shape = {origin_tensor.shape[0], origin_tensor.shape[2], origin_tensor.shape[1]};
      TransLayout(t, context_->GetMemoryManageStreams()[rank_]);
      weights_map_[origin_tensor_name] = t;
    }
    weights_map_.erase(swap_tensor_name);
  }
  return Status();
}

template <typename T>
void Llama4Weight<T>::ProcessWeights() {
  PermuteExpertsWeight();
  CommonMoeWeight<T>::ProcessWeights();
  CommonWeight<T>::ProcessWeights();
}

template class Llama4Weight<float>;
template class Llama4Weight<float16>;
template class Llama4Weight<bfloat16>;

}  // namespace ksana_llm
