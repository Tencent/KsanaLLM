/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/hunyuan_large/hunyuan_large_weight.h"
#include <numeric>

namespace ksana_llm {

template <typename T>
HunyuanLargeWeight<T>::HunyuanLargeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                          int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context),
      CommonMoeWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status HunyuanLargeWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                                  const std::vector<std::string>& weight_name_list,
                                                  const std::vector<std::string>& custom_name_list) {
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  SetDevice(rank_);
  return Status();
}

template <typename T>
Status HunyuanLargeWeight<T>::PermuteQueryWeight(Tensor& last_q_proj_tensor, const int num_layer) {
  if (model_config_.cla_share_factor == 0) return Status();
  SetDevice(rank_);
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    if (layer_idx % model_config_.cla_share_factor == 0) continue;
    std::string gating_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight";
    CommonWeight<T>::CommonPermuteWeight(gating_name, last_q_proj_tensor);
  }
  return Status();
}

template <typename T>
void HunyuanLargeWeight<T>::ProcessWeights() {
  CommonMoeWeight<T>::ProcessWeights();
  int num_layers = model_config_.num_layer;

  // Permute q_proj Weight
  std::string q_proj_name = "model.layers.1.self_attn.q_proj.weight";
  if (num_layers > 1 && weights_map_.find(q_proj_name) != weights_map_.end()) {
    DataType dt = weights_map_[q_proj_name].dtype;
    if (dt == TYPE_FP16 || dt == TYPE_BF16) {
      tensor_manager_->CreateTensorWithSameShape(q_proj_name, "empty_q_proj_tensor");
      Tensor& last_q_proj_tensor = weights_map_["empty_q_proj_tensor"];
      PermuteQueryWeight(last_q_proj_tensor, num_layers);
      weights_map_.erase("empty_q_proj_tensor");
    }
  }

  CommonWeight<T>::ProcessWeights();
}

template class HunyuanLargeWeight<float>;
template class HunyuanLargeWeight<float16>;
template class HunyuanLargeWeight<bfloat16>;

}  // namespace ksana_llm
