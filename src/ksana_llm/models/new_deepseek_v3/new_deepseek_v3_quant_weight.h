/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/kernels/permute.h"

#ifdef ENABLE_CUDA
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
class NewDeepSeekV3QuantWeightBase {
 public:
  virtual ~NewDeepSeekV3QuantWeightBase() = default;
#ifdef ENABLE_FP8
#ifdef ENABLE_FP8_TORCH
  virtual Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(std::unordered_set<std::string> & processed_weights,
                                        std::unordered_set<std::string> & dequant_weights,
                                        int dev_rank,
                                        std::shared_ptr<NewDeepSeekV3Config> &
                                          new_deepseek_v3_config,
                                        std::unordered_map<std::string, Tensor> &
                                          device_model_weights,
                                        std::shared_ptr<Context> & context_) = 0;
#endif
#endif
};

template <typename T>
class NewDeepSeekV3QuantWeight : public NewDeepSeekV3QuantWeightBase {
 public:
  virtual ~NewDeepSeekV3QuantWeight() = default;
#ifdef ENABLE_FP8
#ifdef ENABLE_FP8_TORCH
  Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(std::unordered_set<std::string> & processed_weights,
                                          std::unordered_set<std::string> & dequant_weights,
                                          int dev_rank,
                                          std::shared_ptr<NewDeepSeekV3Config> &
                                            new_deepseek_v3_config,
                                          std::unordered_map<std::string, Tensor> &
                                            device_model_weights,
                                          std::shared_ptr<Context> & context_) override;
#endif
#endif
};

}  // namespace ksana_llm
