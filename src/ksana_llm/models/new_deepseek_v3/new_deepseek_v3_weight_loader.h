/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_quant_weight.h"

namespace ksana_llm {
// The new deepseek loader.
class NewDeepSeekV3WeightLoader : public BaseModelWeightLoader {
 public:
  NewDeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                            std::shared_ptr<Context> context);
  virtual ~NewDeepSeekV3WeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                      std::unordered_map<std::string, Tensor>& device_model_weights,
                                      std::unordered_map<std::string, Tensor>& left_host_weights) override;

 private:
  // local helper functions
  Status TransposeSplitWithFp8Adjustment(const Tensor & host_weight_tensor,
                                          Tensor & output_tensor,
                                          int dev_rank,
                                          std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                          bool is_quant_weight = false);

  Status SplitTransposeWithFp8Adjustment(const Tensor & host_weight_tensor,
                                          Tensor & output_tensor,
                                          int dev_rank,
                                          std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                          bool is_quant_weight = false);
#ifdef ENABLE_FP8
  bool LoadMoeFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights);

  bool LoadMlaFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights);
#endif

  Status GetExpertsIdx(const std::string& expert_name,
                        size_t & layer_idx_,
                        size_t & expert_idx_);

  Status ProcessGateUpProjWeight(std::string& file_weight_name_,
                                      const Tensor& dev_tensor,
                                      std::unordered_map<std::string, Tensor>& device_model_weights,
                                      int dev_rank,
                                      std::unordered_set<std::string>& processed_weights);

  Status InitQuantWeightLoader(std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config);

 private:
  PipelineConfig pipeline_config_;
  std::unique_ptr<NewDeepSeekV3QuantWeightBase> quant_weight_;
};
}  // namespace ksana_llm
