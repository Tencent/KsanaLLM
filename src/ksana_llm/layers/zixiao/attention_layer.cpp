/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "AttentionLayer not supported.");
}

template class AttentionLayer<float>;
template class AttentionLayer<float16>;
template class AttentionLayer<bfloat16>;

}  // namespace ksana_llm
