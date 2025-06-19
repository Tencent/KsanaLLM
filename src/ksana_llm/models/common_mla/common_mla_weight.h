/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class CommonMlaWeight : virtual public CommonWeight<T> {
 public:
  CommonMlaWeight() {}
  explicit CommonMlaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list,
                             std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::context_;
  using BaseWeight::rank_;
  using CommonWeight<T>::tensor_para_size_;

  using CommonWeight<T>::moe_weight_data_type_;
  using BaseWeight::weights_data_type_map_;
  using BaseWeight::weights_map_;

  using BaseWeight::model_config_;

  using CommonWeight<T>::tensor_manager_;

  using CommonWeight<T>::quant_weight_solver_;

  using BaseWeight::pipeline_config_;
  using CommonWeight<T>::required_layer_idx_;

 private:
  Status PermuteQaWeight(Tensor& last_q_a_proj_tensor, bool is_weight_scale);
  Status PermuteQbNopeWeight(Tensor& last_q_b_nope_tensor, bool is_weight_scale);
  Status PermuteQbRopeWeight(Tensor& last_q_b_rope_tensor, bool is_weight_scale);
  Status PermuteKVaLoraWeight(Tensor& last_kv_a_lora_tensor, bool is_weight_scale);
  Status PermuteKVaRopeWeight(Tensor& last_kv_a_rope_tensor, bool is_weight_scale);
  Status PermuteKVbNopeWeight(Tensor& last_kv_b_nope_tensor, bool is_weight_scale);
  Status PermuteVHeadWeight(Tensor& last_v_head_tensor, bool is_weight_scale);
  Status PermuteMlaWeight(bool is_weight_scale);
};

}  // namespace ksana_llm