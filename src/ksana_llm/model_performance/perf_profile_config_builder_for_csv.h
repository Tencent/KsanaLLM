/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/model_performance/model_performance_runner.h"

namespace ksana_llm {

class PerfProfileConfigBuilderWithCsv : public PerfProfileConfigBuilderInterface {
 public:
  PerfProfileConfigBuilderWithCsv(const std::string& csv_filename, size_t warmup_round, size_t profile_round);

  PerfProfileConfig GetMaxPerfProfileConfig() override;

  void GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs) override;

 private:
  Status ParsePerformanceRunnerConfig(const std::string& config_file);

 private:
  std::vector<PerfProfileConfig> csv_configs_;
  // runner_config
  size_t warmup_round_;
  size_t profile_round_;
};

}  // namespace ksana_llm
