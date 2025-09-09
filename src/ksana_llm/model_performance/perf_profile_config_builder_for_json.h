/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/model_performance/model_performance_runner.h"

namespace ksana_llm {

class PerfProfileConfigBuilderWithJson : public PerfProfileConfigBuilderInterface {
 public:
  PerfProfileConfigBuilderWithJson(const std::string& json_filename, size_t warmup_round, size_t profile_round,
                                   bool same_batch_in_dps = false);

  PerfProfileConfig GetMaxPerfProfileConfig() override;

  void GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs, size_t attn_dp_num) override;

 private:
  Status ParsePerformanceRunnerConfig(const std::string& config_file);
  void ValidateDpConfigs(const std::vector<PerfProfileConfig>& configs, size_t attn_dp_num);

 private:
  std::vector<PerfProfileConfig> json_configs_;
  std::string config_file_path_;  // Store config file path
  // runner_config
  size_t warmup_round_;
  size_t profile_round_;
  bool same_batch_in_dps_;  // 标志是否在所有DP中使用相同的batch配置
};

}  // namespace ksana_llm