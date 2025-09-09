/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/model_performance/perf_profile_config_builder_for_json.h"

#include <sstream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>


namespace ksana_llm {

PerfProfileConfigBuilderWithJson::PerfProfileConfigBuilderWithJson(const std::string& json_filename,
                                                                   size_t warmup_round, size_t profile_round,
                                                                   bool same_batch_in_dps)
    : warmup_round_(warmup_round),
      profile_round_(profile_round),
      config_file_path_(json_filename),
      same_batch_in_dps_(same_batch_in_dps) {
  // Parse the JSON configuration file immediately
  auto status = ParsePerformanceRunnerConfig(json_filename);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Failed to parse JSON config file: " << status.GetMessage();
    KLLM_THROW(FormatStr("Failed to parse JSON config file: %s", status.GetMessage().c_str()));
  }
}

PerfProfileConfig PerfProfileConfigBuilderWithJson::GetMaxPerfProfileConfig() {
  if (json_configs_.empty()) {
    // Return an empty config if no configs were loaded
    PerfProfileConfig empty_config;
    empty_config.config_id = 0;
    empty_config.warmup_round = warmup_round_;
    empty_config.profile_round = profile_round_;
    empty_config.req_configs.resize(1);
    return empty_config;
  }

  // Find the config with the maximum total tokens
  PerfProfileConfig max_config = json_configs_[0];
  size_t max_value = 0;

  for (const auto& config : json_configs_) {
    for (const auto& req_config : config.req_configs) {
      size_t total_tokens = 0;
      for (const auto& req : req_config.reqs) {
        total_tokens += req.sequence_len * req.request_num;
      }

      if (total_tokens > max_value) {
        max_value = total_tokens;
        max_config = config;
      }
    }
  }

  return max_config;
}

void PerfProfileConfigBuilderWithJson::GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs,
                                                             size_t attn_dp_num) {
  configs = json_configs_;

  // If same_batch_in_dps is enabled, make all DP nodes use the same configuration as dp_0
  if (same_batch_in_dps_) {
    for (auto& config : configs) {
      if (!config.req_configs.empty()) {
        // Use dp_0's configuration for all DP nodes
        auto dp0_config = config.req_configs[0];
        config.req_configs.resize(attn_dp_num);
        for (size_t i = 0; i < attn_dp_num; ++i) {
          config.req_configs[i] = dp0_config;
        }
      }
    }
  }

  ValidateDpConfigs(configs, attn_dp_num);
}

void PerfProfileConfigBuilderWithJson::ValidateDpConfigs(const std::vector<PerfProfileConfig>& configs,
                                                         size_t attn_dp_num) {
  for (const auto& config : configs) {
    if (config.req_configs.size() != attn_dp_num) {
      KLLM_LOG_ERROR << "Configuration DP count mismatch: config has " << config.req_configs.size()
                     << " DPs but runtime expects " << attn_dp_num;
      KLLM_THROW(FormatStr("Configuration DP count mismatch: config has %zu DPs but runtime expects %zu",
                           config.req_configs.size(), attn_dp_num));
    }
  }
}

Status PerfProfileConfigBuilderWithJson::ParsePerformanceRunnerConfig(const std::string& config_file) {
  std::ifstream file(config_file);
  if (!file.is_open()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Failed to open JSON file: " + config_file);
  }

  nlohmann::json json_data;
  try {
    file >> json_data;
  } catch (const std::exception& e) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Failed to parse JSON file: " + config_file + ", error: " + e.what());
  }

  // Parse global layer_forward_round if present
  size_t layer_forward_round = 1;  // Default value
  if (json_data.contains("layer_forward_round")) {
    layer_forward_round = json_data["layer_forward_round"];
    // Remove it from json_data so it's not processed as a batch
    json_data.erase("layer_forward_round");
  }

  uint32_t config_id = 0;
  for (auto& [batch_name, batch_data] : json_data.items()) {
    // Create a new PerfProfileConfig for each batch
    PerfProfileConfig config;
    config.config_id = config_id++;
    config.warmup_round = warmup_round_;
    config.profile_round = profile_round_;
    config.layer_forward_round = layer_forward_round;

    // Only support object format: "batch_0": { "dp_0": [...], "dp_1": [...] }
    if (batch_data.is_object()) {
      // Format: "batch_0": { "dp_0": [...], "dp_1": [...] }
      // Lambda function to parse request info from JSON
      auto parseRequestInfo = [&](const nlohmann::json& req,
                                  const std::string& context) -> PerfProfileConfig::PerfDpReqConfig::RequestInfo {
        if (!req.contains("forwarding_token_num") || !req.contains("sequence_len")) {
          KLLM_THROW(
              FormatStr("Invalid JSON format: missing 'forwarding_token_num' or 'sequence_len' in %s of batch %s",
                        context.c_str(), batch_name.c_str()));
        }
        PerfProfileConfig::PerfDpReqConfig::RequestInfo req_info;
        req_info.forwarding_token_num = req["forwarding_token_num"];
        req_info.sequence_len = req["sequence_len"];
        if (req.contains("request_num")) {
          req_info.request_num = req["request_num"];
        }
        return req_info;
      };

      // Parse different configurations for different DP nodes (always parse all available DPs)
      int max_dp_idx = -1;
      for (auto& [dp_name, dp_requests] : batch_data.items()) {
        if (dp_name.substr(0, 3) != "dp_") {
          return Status(RetCode::RET_INVALID_ARGUMENT,
                        "Invalid JSON format: expected 'dp_X' key, got '" + dp_name + "' in batch " + batch_name);
        }
        max_dp_idx = std::max(max_dp_idx, std::stoi(dp_name.substr(3)));
      }
      config.req_configs.resize(max_dp_idx + 1);
      for (auto& [dp_name, dp_requests] : batch_data.items()) {
        int dp_idx = std::stoi(dp_name.substr(3));
        std::vector<PerfProfileConfig::PerfDpReqConfig::RequestInfo> reqs;
        for (const auto& req : dp_requests) {
          reqs.push_back(parseRequestInfo(req, dp_name));
        }
        config.req_configs[dp_idx].reqs = reqs;
      }
    } else {
      return Status(RetCode::RET_INVALID_ARGUMENT,
                    "Invalid JSON format: batch " + batch_name + " should be an object or array");
    }

    // Add validation for layer_forward_round similar to CSV parser
    KLLM_CHECK_WITH_INFO(config.layer_forward_round < 100,
                         FormatStr("config.layer_forward_round==%d, must <= 100.", config.layer_forward_round));

    json_configs_.push_back(config);
  }

  if (json_configs_.empty()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "No valid configurations found in JSON file");
  }

  return Status();
}

}  // namespace ksana_llm