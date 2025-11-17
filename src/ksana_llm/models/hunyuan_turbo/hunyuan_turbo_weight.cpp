/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/hunyuan_turbo/hunyuan_turbo_weight.h"
#include <numeric>

namespace ksana_llm {

template <typename T>
HunyuanTurboWeight<T>::HunyuanTurboWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                          int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status HunyuanTurboWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                                  const std::vector<std::string>& weight_name_list,
                                                  const std::vector<std::string>& custom_name_list) {
  CommonWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  SetDevice(rank_);
  return Status();
}
template <typename T>
void HunyuanTurboWeight<T>::ProcessWeights() {
  CommonWeight<T>::ProcessWeights();
  CommonWeight<T>::PrintDebugMessage();
}

template class HunyuanTurboWeight<float>;
template class HunyuanTurboWeight<float16>;
template class HunyuanTurboWeight<bfloat16>;

}  // namespace ksana_llm
