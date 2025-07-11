/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/qwen3_moe/qwen3_moe_weight.h"

namespace ksana_llm {

template <typename T>
Qwen3MoeWeight<T>::Qwen3MoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context), CommonMoeWeight<T>(model_config, rank, context) {}

template <typename T>
Status Qwen3MoeWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                              const std::vector<std::string>& weight_name_list,
                                              const std::vector<std::string>& custom_name_list) {
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  return Status();
}

template <typename T>
void Qwen3MoeWeight<T>::ProcessWeights() {
  CommonMoeWeight<T>::ProcessWeights();
  CommonWeight<T>::ProcessWeights();
}

template <typename T>
void Qwen3MoeWeight<T>::SetEmbeddingsConfig() {
  CommonWeight<T>::SetEmbeddingsConfig();
}

template class Qwen3MoeWeight<float>;
template class Qwen3MoeWeight<float16>;
template class Qwen3MoeWeight<bfloat16>;

}  // namespace ksana_llm