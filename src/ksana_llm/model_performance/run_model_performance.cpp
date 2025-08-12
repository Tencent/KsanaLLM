/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"

void Usage() {
  std::cout << "Usage: ./run_model_performance config_file model_config_name" << std::endl;
  std::cout << "Default value of model_config_name: config.json" << std::endl;
  std::cout << "Example 1: ./run_model_performance llama_7b_performance_run.yaml" << std::endl;
  std::cout << "Example 2: ./run_model_performance deepseek_r1_performance_run.yaml config_6layers.json"
            << std::endl;
}

// usage: run_model_performance /data/llama/ksana_config.yaml
int main(int argc, char *argv[]) {
  std::string config_path;
  std::string model_config_filename = "config.json";
  if (argc >= 3) {
    model_config_filename = argv[2];
    std::cout << "using config filename: " << model_config_filename << std::endl;
  }
  if (argc >= 2) {
    config_path = argv[1];
    std::cout << "using config: " << config_path << std::endl;
  } else if (argc == 1) {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_performance_run.yaml";
    config_path = std::filesystem::absolute(config_path_relate).string();
    std::cout << "using demo config: " << config_path << std::endl;
  } else {
    std::cout << "Invalid number of arguments. Please provide the path of one config" << std::endl;
    Usage();
    return 1;
  }

  std::shared_ptr<ksana_llm::ModelPerformanceRunner> model_performance_runner =
      std::make_shared<ksana_llm::ModelPerformanceRunner>(config_path, model_config_filename);
  model_performance_runner->RunPerformanceForward();
  return 0;
}
