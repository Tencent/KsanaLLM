/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"
#include "ksana_llm/model_performance/perf_profile_config_builder_for_csv.h"

using namespace ksana_llm;

void Usage() {
  std::cout << "Usage: ./run_model_performance [OPTIONS]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --csv-config <file>    Specify profile configuration file in CSV format" << std::endl;
  std::cout << "  --runtime-config <file>  Specify runtime config filename" << std::endl;
  std::cout << "  --warmup-round <num>   Number of warmup rounds (default: 2)" << std::endl;
  std::cout << "  --profile-round <num>  Number of profile rounds (default: 100)" << std::endl;
  std::cout << "  --output <file>        Output results to CSV file" << std::endl;
  std::cout << "  --lower-layer-idx <num> Lower layer index for partial model execution (default: -1)" << std::endl;
  std::cout << "  --upper-layer-idx <num> Upper layer index for partial model execution (default: -1)" << std::endl;
  std::cout << "Examples:" << std::endl;
  std::cout << "  ./run_model_performance --runtime-config llama_7b_performance_run.yaml --csv-config "
               "perf_profile_configs.csv "
            << std::endl;
  std::cout << "  ./run_model_performance --runtime-config llama_7b_performance_run.yaml --csv-config "
               "perf_profile_configs.csv --warmup-round 5 --profile-round 50"
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

  // Parse command line arguments
  if (argc == 1) {
    Usage();

    // No arguments provided, use default config
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_tp.yaml";
    runtime_config_path = std::filesystem::absolute(config_path_relate).string();
    std::filesystem::path profile_csv_config_relate = parent_path / "test_config.csv";
    perf_config_path = std::filesystem::absolute(profile_csv_config_relate).string();
    warmup_round = 10;
    profile_round = 100;
    std::cout << "No arguments provided. Using demo perf config: " << perf_config_path
              << ", runtime config: " << runtime_config_path << std::endl;
  } else {
    // Parse named arguments
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--csv-config" && i + 1 < argc) {
        perf_config_path = argv[++i];
        std::cout << "Using CSV config: " << perf_config_path << std::endl;
      } else if (arg == "--runtime-config" && i + 1 < argc) {
        runtime_config_path = argv[++i];
        std::cout << "Using runtime config: " << runtime_config_path << std::endl;
      } else if (arg == "--warmup-round" && i + 1 < argc) {
        try {
          warmup_round = std::stoull(argv[++i]);
          std::cout << "Using warmup round: " << warmup_round << std::endl;
        } catch (const std::exception& e) {
          std::cout << "Invalid warmup round value: " << argv[i] << std::endl;
          Usage();
          return 1;
        }
      } else if (arg == "--profile-round" && i + 1 < argc) {
        try {
          profile_round = std::stoull(argv[++i]);
          std::cout << "Using profile round: " << profile_round << std::endl;
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
      } else {
        std::cout << "Invalid argument: " << arg << std::endl;
        Usage();
        return 1;
      }
    }

    // Check if config path was provided
    if (perf_config_path.empty()) {
      std::cout << "Error: No perf configuration file specified." << std::endl;
      Usage();
      return 1;
    }

    if (runtime_config_path.empty()) {
      std::cout << "Error: No runtime configuration file specified." << std::endl;
      Usage();
      return 1;
    }
  }

  // Initialize the appropriate config builder based on the config type
  std::shared_ptr<PerfProfileConfigBuilderInterface> config_builder =
      std::make_shared<ksana_llm::PerfProfileConfigBuilderWithCsv>(perf_config_path, warmup_round, profile_round);

  std::shared_ptr<ksana_llm::ModelPerformanceRunner> model_performance_runner =
      std::make_shared<ksana_llm::ModelPerformanceRunner>(
          runtime_config_path, config_builder->GetMaxPerfProfileConfig(), lower_layer_idx, upper_layer_idx);
  config_builder->SetAttnDpNum(model_performance_runner->GetAttnDpNum());
  std::vector<PerfProfileConfig> configs;
  config_builder->GetPerfProfileConfigs(configs);

  // Open output file if specified
  std::ofstream output_file;
  if (!output_file_path.empty()) {
    output_file.open(output_file_path);
    if (!output_file.is_open()) {
      std::cout << "Failed to open output file: " << output_file_path << std::endl;
      return 1;
    }
    // Write CSV header
    output_file << "single_token_request_num,single_token_request_cached_token_num,"
                << "multi_token_request_num,multi_token_request_token_num,"
                << "profile_round,time_cost_ms,avg_time_ms_per_round" << std::endl;
  }
  for (auto& config : configs) {
    PerfProfileResult result;
    auto status = model_performance_runner->RunPerformanceForward(config, result);
    std::stringstream ss;
    if (status.OK()) {
      auto& req_config = config.req_configs[0];
      ss << fmt::format(
                "\n single_token_request_num:{}, "
                "single_token_request_cached_token_num:{}, "
                "multi_token_request_num:{}, multi_token_request_token_num:{}, "
                "Performance Results: {} rounds cost {} milliseconds,",
                req_config.single_token_request_num, req_config.single_token_request_cached_token_num,
                req_config.multi_token_request_num, req_config.multi_token_request_token_num, config.profile_round,
                result.time_cost_ms)
         << fmt::format("Average: {} milliseconds/round \n", result.time_cost_ms / config.profile_round);
      KLLM_LOG_INFO << ss.str();
    } else {
      ss << fmt::format("Faild to run model_preformance. End with status {}", status.GetMessage());
      KLLM_LOG_ERROR << ss.str();
    }
    std::cout << ss.str();

    // Write to CSV file if specified
    if (output_file.is_open() && status.OK()) {
      // TODO(robertyuan): change this if dps have different loads.
      auto& req_config = config.req_configs[0];
      output_file << req_config.single_token_request_num << "," << req_config.single_token_request_cached_token_num
                  << "," << req_config.multi_token_request_num << "," << req_config.multi_token_request_token_num << ","
                  << config.profile_round << "," << result.time_cost_ms << ","
                  << (result.time_cost_ms / config.profile_round) << std::endl;
    }
  }

  // Close output file if opened
  if (output_file.is_open()) {
    output_file.close();
  }
  return 0;
}
