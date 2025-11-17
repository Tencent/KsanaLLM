/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#include <iomanip>
#include "ksana_llm/model_performance/model_performance_runner.h"
#include "ksana_llm/model_performance/perf_profile_config_builder_for_json.h"
#include "ksana_llm/utils/environment.h"

using namespace ksana_llm;

void Usage() {
  std::cout << "Usage: ./run_model_performance [OPTIONS]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --perf-config <file>   Specify profile configuration file in JSON format" << std::endl;
  std::cout << "  --runtime-config <file>  Specify runtime config filename" << std::endl;
  std::cout << "  --warmup-round <num>   Number of warmup rounds (default: 2)" << std::endl;
  std::cout << "  --profile-round <num>  Number of profile rounds (default: 100)" << std::endl;
  std::cout << "  --output <file>        Output results to CSV file" << std::endl;
  std::cout << "  --lower-layer-idx <num> Lower layer index for partial model execution (default: -1)" << std::endl;
  std::cout << "  --upper-layer-idx <num> Upper layer index for partial model execution (default: -1)" << std::endl;
  std::cout << "  --same-batch-in-dps    Use same batch configuration for all DP nodes (default: false)" << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  ./run_model_performance --runtime-config llama_7b_performance_run.yaml --perf-config "
               "test_config.json --warmup-round 5 --profile-round 50 --same-batch-in-dps"
            << std::endl;
}

// usage: run_model_performance /data/llama/ksana_config.yaml
int main(int argc, char* argv[]) {
  InitLoguru();

  // Default values
  std::string perf_config_path;

  std::string runtime_config_path;

  // Default values for warmup and profile rounds
  size_t warmup_round = 2;
  size_t profile_round = 100;

  // Output file path (empty if not specified)
  std::string output_file_path;

  // Default values for layer indices
  int16_t lower_layer_idx = -1;
  int16_t upper_layer_idx = -1;

  // Default value for same_batch_in_dps flag
  bool same_batch_in_dps = false;

  // Parse command line arguments
  if (argc == 1) {
    Usage();

    // No arguments provided, use default config
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_tp.yaml";
    runtime_config_path = std::filesystem::absolute(config_path_relate).string();
    std::filesystem::path profile_json_config_relate = parent_path / "test_config.json";
    perf_config_path = std::filesystem::absolute(profile_json_config_relate).string();
    warmup_round = 10;
    profile_round = 100;
    std::cout << "No arguments provided. Using demo perf config: " << perf_config_path
              << ", runtime config: " << runtime_config_path << std::endl;
  } else {
    // Parse named arguments
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--perf-config" && i + 1 < argc) {
        perf_config_path = argv[++i];
        std::cout << "Using JSON config: " << perf_config_path << std::endl;
      } else if (arg == "--runtime-config" && i + 1 < argc) {
        runtime_config_path = argv[++i];
        std::cout << "Using runtime config: " << runtime_config_path << std::endl;
      } else if (arg == "--warmup-round" && i + 1 < argc) {
        try {
          warmup_round = std::stoull(argv[++i]);
        } catch (const std::exception& e) {
          std::cout << "Invalid warmup round value: " << argv[i] << std::endl;
          Usage();
          return 1;
        }
      } else if (arg == "--profile-round" && i + 1 < argc) {
        try {
          profile_round = std::stoull(argv[++i]);
        } catch (const std::exception& e) {
          std::cout << "Invalid profile round value: " << argv[i] << std::endl;
          Usage();
          return 1;
        }
      } else if (arg == "--output" && i + 1 < argc) {
        output_file_path = argv[++i];
        std::cout << "Using output file: " << output_file_path << std::endl;
      } else if (arg == "--lower-layer-idx" && i + 1 < argc) {
        try {
          lower_layer_idx = static_cast<int16_t>(std::stoi(argv[++i]));
          std::cout << "Using lower layer index: " << lower_layer_idx << std::endl;
        } catch (const std::exception& e) {
          std::cout << "Invalid lower layer index value: " << argv[i] << std::endl;
          Usage();
          return 1;
        }
      } else if (arg == "--upper-layer-idx" && i + 1 < argc) {
        try {
          upper_layer_idx = static_cast<int16_t>(std::stoi(argv[++i]));
          std::cout << "Using upper layer index: " << upper_layer_idx << std::endl;
        } catch (const std::exception& e) {
          std::cout << "Invalid upper layer index value: " << argv[i] << std::endl;
          Usage();
          return 1;
        }
      } else if (arg == "--same-batch-in-dps") {
        same_batch_in_dps = true;
        std::cout << "Using same batch configuration for all DP nodes" << std::endl;
      } else {
        std::cout << "Invalid argument: " << arg << std::endl;
        Usage();
        return 1;
      }
    }

    // Check if config path was provided
    if (perf_config_path.empty()) {
      std::cout << "Error: No perf configuration file specified (use --perf-config)." << std::endl;
      Usage();
      return 1;
    }

    if (runtime_config_path.empty()) {
      std::cout << "Error: No runtime configuration file specified." << std::endl;
      Usage();
      return 1;
    }
  }

  // Initialize the JSON config builder
  std::shared_ptr<PerfProfileConfigBuilderInterface> config_builder;
  config_builder = std::make_shared<ksana_llm::PerfProfileConfigBuilderWithJson>(perf_config_path, warmup_round,
                                                                                 profile_round, same_batch_in_dps);

  std::shared_ptr<ksana_llm::ModelPerformanceRunner> model_performance_runner =
      std::make_shared<ksana_llm::ModelPerformanceRunner>(
          runtime_config_path, config_builder->GetMaxPerfProfileConfig(), lower_layer_idx, upper_layer_idx);
  std::cout << "Detect DP Size = " << model_performance_runner->GetAttnDpNum() << std::endl;
  std::vector<PerfProfileConfig> configs;
  config_builder->GetPerfProfileConfigs(configs, model_performance_runner->GetAttnDpNum());

  // Open output file if specified
  std::ofstream output_file;
  if (!output_file_path.empty()) {
    output_file.open(output_file_path);
    if (!output_file.is_open()) {
      std::cout << "Failed to open output file: " << output_file_path << std::endl;
      return 1;
    }
    // Write CSV header with grouped prefill/decode fields using fixed width formatting
    // Write CSV header with grouped prefill/decode fields (space before decode group for visual separation)
    output_file << "profile_round,layer_forward_round,total_requests,total_sequence_len,"
                << "prefill_requests,prefill_avg_seq_len,prefill_avg_forwarding_tokens,"
                << " decode_requests,decode_avg_seq_len,decode_avg_forwarding_tokens,"
                << "avg_time_ms_per_profile_round" << std::endl;
  }
  for (auto& config : configs) {
    PerfProfileResult result;
    std::stringstream ss;
    ss << "Running config: " << config.ToStr();
    KLLM_LOG_INFO << ss.str();
    std::cout << ss.str() << std::endl;
    auto status = model_performance_runner->RunPerformanceForward(config, result);
    ss.clear();
    ss.str("");
    if (status.OK()) {
      ss << fmt::format("Results: {} rounds cost {} milliseconds, average: {} milliseconds/round", config.profile_round,
                        result.time_cost_ms, result.time_cost_ms / config.profile_round);
      KLLM_LOG_INFO << ss.str();
    } else {
      ss << fmt::format("  Faild to run model_preformance. End with status {}", status.GetMessage());
      KLLM_LOG_ERROR << ss.str();
    }
    std::cout << ss.str() << std::endl;

    // Write to CSV file if specified
    if (output_file.is_open() && status.OK()) {
      // Write one line for each DP configuration
      for (const auto& req_config : config.req_configs) {
        size_t total_requests = 0;
        size_t prefill_requests = 0;
        size_t decode_requests = 0;
        size_t total_sequence_len = 0;

        // Variables for calculating averages
        size_t prefill_total_seq_len = 0;
        size_t decode_total_seq_len = 0;
        size_t prefill_total_forwarding_tokens = 0;
        size_t decode_total_forwarding_tokens = 0;

        for (const auto& req : req_config.reqs) {
          // Consider request_num when calculating totals
          total_requests += req.request_num;
          total_sequence_len += req.sequence_len * req.request_num;

          if (req.forwarding_token_num <= GetDecodeTokenNumThreshold()) {
            decode_requests += req.request_num;
            decode_total_seq_len += req.sequence_len * req.request_num;
            decode_total_forwarding_tokens += req.forwarding_token_num * req.request_num;
          } else {
            prefill_requests += req.request_num;
            prefill_total_seq_len += req.sequence_len * req.request_num;
            prefill_total_forwarding_tokens += req.forwarding_token_num * req.request_num;
          }
        }

        // Calculate averages (avoid division by zero)
        double prefill_avg_seq_len =
            prefill_requests > 0 ? static_cast<double>(prefill_total_seq_len) / prefill_requests : 0.0;
        double decode_avg_seq_len =
            decode_requests > 0 ? static_cast<double>(decode_total_seq_len) / decode_requests : 0.0;
        double prefill_avg_forwarding_tokens =
            prefill_requests > 0 ? static_cast<double>(prefill_total_forwarding_tokens) / prefill_requests : 0.0;
        double decode_avg_forwarding_tokens =
            decode_requests > 0 ? static_cast<double>(decode_total_forwarding_tokens) / decode_requests : 0.0;

        output_file << config.profile_round << "," << config.layer_forward_round << "," << total_requests << ","
                    << total_sequence_len << "," << prefill_requests << "," << std::fixed << std::setprecision(3)
                    << prefill_avg_seq_len << "," << std::fixed << std::setprecision(3) << prefill_avg_forwarding_tokens
                    << "," << " " << decode_requests << "," << std::fixed << std::setprecision(3) << decode_avg_seq_len
                    << "," << std::fixed << std::setprecision(3) << decode_avg_forwarding_tokens << "," << std::fixed
                    << std::setprecision(3) << (result.time_cost_ms / config.profile_round) << std::endl;
      }
    }
  }

  // Close output file if opened
  if (output_file.is_open()) {
    output_file.close();
  }
  return 0;
}
