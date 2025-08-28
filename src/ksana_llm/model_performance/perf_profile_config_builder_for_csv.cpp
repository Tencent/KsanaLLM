/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/model_performance/perf_profile_config_builder_for_csv.h"

#include <fstream>
#include <sstream>
#include <string>

namespace ksana_llm {

PerfProfileConfigBuilderWithCsv::PerfProfileConfigBuilderWithCsv(const std::string& config_file, size_t warmup_round,
                                                                 size_t profile_round)
    : warmup_round_(warmup_round), profile_round_(profile_round) {
  Status status = ParsePerformanceRunnerConfig(config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Error: " << status.GetMessage();
  }
}

PerfProfileConfig PerfProfileConfigBuilderWithCsv::GetMaxPerfProfileConfig() {
  if (csv_configs_.empty()) {
    // Return an empty config if no configs were loaded
    PerfProfileConfig empty_config;
    empty_config.config_id = 0;
    empty_config.warmup_round = warmup_round_;
    empty_config.profile_round = profile_round_;
    empty_config.req_configs.resize(1);
    return empty_config;
  }

  // Find the config with the maximum value of:
  // single_token_request_num * single_token_request_cached_token_num +
  // multi_token_request_num * multi_token_request_token_num
  PerfProfileConfig max_config = csv_configs_[0];
  size_t max_value = 0;

  for (const auto& config : csv_configs_) {
    for (const auto& req_config : config.req_configs) {
      size_t current_value = req_config.single_token_request_num * req_config.single_token_request_cached_token_num +
                             req_config.multi_token_request_num * req_config.multi_token_request_token_num;

      if (current_value > max_value) {
        max_value = current_value;
        max_config = config;
      }
    }
  }

  return max_config;
}

void PerfProfileConfigBuilderWithCsv::GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs) {
  if (!csv_config_dp_initialized_) {
    // TODO(robertyuan): Support different config for dp
    for (auto& config : csv_configs_) {
      auto req_config = config.req_configs[0];
      config.req_configs.resize(dp_num_);
      for (size_t dp_idx = 1; dp_idx < dp_num_; dp_idx++) {
        config.req_configs[dp_idx] = req_config;
      }
    }
    csv_config_dp_initialized_ = true;
  }
  configs = csv_configs_;
}

Status PerfProfileConfigBuilderWithCsv::ParsePerformanceRunnerConfig(const std::string& config_file) {
  std::ifstream file(config_file);
  if (!file.is_open()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Failed to open CSV file: " + config_file);
  }

  // Read header line to verify column structure
  std::string header_line;
  if (!std::getline(file, header_line)) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "CSV file is empty: " + config_file);
  }

  // Parse each data row
  const size_t kCsvColumnNum = 6;  // Expected number of columns
  std::string line;
  uint32_t config_id = 0;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;  // Skip empty lines
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<size_t> values;

    // Parse CSV row
    while (std::getline(ss, cell, ',')) {
      try {
        size_t value = std::stoull(cell);
        values.push_back(value);
      } catch (const std::exception& e) {
        return Status(RetCode::RET_INVALID_ARGUMENT, "Failed to parse CSV value: " + cell + ", error: " + e.what());
      }
    }

    // Verify we have the expected number of columns
    if (values.size() < kCsvColumnNum) {
      return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid CSV format, expected at least " +
                                                       std::to_string(kCsvColumnNum) + " columns, got " +
                                                       std::to_string(values.size()));
    }

    // Create a new PerfProfileConfig
    PerfProfileConfig config;
    config.config_id = config_id++;
    config.warmup_round = warmup_round_;
    config.profile_round = profile_round_;
    config.req_configs.resize(1);

    // Fill in the values from the CSV row
    config.req_configs[0].single_token_request_num = values[0];
    config.req_configs[0].single_token_request_cached_token_num = values[1];
    config.req_configs[0].multi_token_request_num = values[2];
    config.req_configs[0].multi_token_request_token_num = values[3];
    config.req_configs[0].multi_token_forwarding_token_num = values[4];

    config.layer_forward_round = values[5];
    KLLM_CHECK_WITH_INFO(config.layer_forward_round < 100,
                         FormatStr("config.layer_forward_round==%d, must <= 100.", config.layer_forward_round));

    // Add the config to our list
    csv_configs_.push_back(config);
  }

  if (csv_configs_.empty()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "No valid configurations found in CSV file");
  }

  return Status();
}

}  // namespace ksana_llm
