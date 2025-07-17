/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/distributed/distributed_coordinator.h"
#include "ksana_llm/runtime/llm_runtime.h"

namespace ksana_llm {

class CommunicationPerformanceRunner {
 public:
  explicit CommunicationPerformanceRunner(const std::string& config_path);

  ~CommunicationPerformanceRunner();

  void Run();

 private:
  void InitEnvs(const std::string& config_path);

  void TestCommunicatePerformance(const std::vector<size_t>& shape, DataType data_type);

 private:
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<DistributedCoordinator> distributed_coordinator_ = nullptr;
};
}  // namespace ksana_llm
