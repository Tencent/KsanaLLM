/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
class FlashMlaAttentionLayer : public AttentionLayer<SCALAR_T> {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

#ifdef ENABLE_ACL
 private:
  // ATB attention implement for ascend device, for ATB run with graph, each layer need's one kernel implement instance
  std::shared_ptr<llm_kernels::ascend::ATBAttention<SCALAR_T>> atb_flash_attn_;
#endif
};

}  // namespace ksana_llm
