/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/mixtral/mixtral_weight.h"

namespace ksana_llm {

template <typename T>
MixtralWeight<T>::MixtralWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context), CommonMoeWeight<T>(model_config, rank, context) {}

template <typename T>
Status MixtralWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                                             std::vector<std::string>& weight_name_list,
                                             std::vector<std::string>& custom_name_list) {
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  return Status();
}

template <typename T>
void MixtralWeight<T>::ProcessWeights() {
  CommonMoeWeight<T>::ProcessWeights();
  CommonWeight<T>::ProcessWeights();
}

template class MixtralWeight<float>;
template class MixtralWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class MixtralWeight<bfloat16>;
#endif

}  // namespace ksana_llm
