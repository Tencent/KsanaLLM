/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "ksana_llm/model_performance/communication_performance_runner.h"

// usage: run_model_communication /data/llama/ksana_config.yaml
int main(int argc, char *argv[]) {
  std::string config_path;
  if (argc == 2) {
    config_path = argv[1];
    std::cout << "using config: " << config_path << std::endl;
  } else if (argc == 1) {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    config_path = std::filesystem::absolute(config_path_relate).string();
    std::cout << "using demo config: " << config_path << std::endl;
  } else {
    std::cout << "Invalid number of arguments. Please provide the path of one config" << std::endl;
    return 1;
  }

  std::shared_ptr<ksana_llm::CommunicationPerformanceRunner> communication_performance_runner =
      std::make_shared<ksana_llm::CommunicationPerformanceRunner>(config_path);
  communication_performance_runner->Run();
  return 0;
}