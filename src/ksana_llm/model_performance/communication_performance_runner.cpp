/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/model_performance/communication_performance_runner.h"

#include <algorithm>
#include <filesystem>
#include <random>
#include <thread>

#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/distributed_coordinator.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace ksana_llm {

CommunicationPerformanceRunner::CommunicationPerformanceRunner(const std::string& config_path) {
  InitEnvs(config_path);
}

CommunicationPerformanceRunner::~CommunicationPerformanceRunner() {
  distributed_coordinator_->DestroyCluster();
  distributed_coordinator_ = nullptr;
  DestroyScheduleOutputPool();
  if (!context_->IsStandalone()) {
    DestroyHiddenUnitBufferPool();
  }
}

void CommunicationPerformanceRunner::InitEnvs(const std::string& config_path) {
  InitLoguru();

  const auto& env = Singleton<Environment>::GetInstance();
  env->ParseConfig(config_path);

  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  pipeline_config.SetDistributeRelatedConfig();
  Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);

  // init context
  constexpr int max_multi_batch_num = 1;
  context_.reset(new Context(env->GetTensorParallelSize(), env->GetAttnDataParallelSize(), max_multi_batch_num));
  KLLM_CHECK_WITH_INFO(!context_->IsStandalone(), "Failed to get batch scheduler config error");

  // init model_config
  env->GetModelConfig("", model_config_);

  InitializeScheduleOutputPool();
  InitializeHiddenUnitBufferPool();

  // init DistributedCoordinator
  distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
      context_, GetPacketObject, GetScheduleOutputPool(), GetHiddenUnitBufferPool(), nullptr, env);
  KLLM_LOG_INFO << "Initialize distributed coordinator.";
  distributed_coordinator_->InitializeCluster();
  KLLM_LOG_INFO << "Start to synchronize node layers.";
  distributed_coordinator_->SynchronizeNodeLayers(0);

  GetHiddenUnitBufferPool()->PreAllocateDeviceBuffer();
}

void CommunicationPerformanceRunner::Run() {
  size_t test_token_num = 1;
  while (test_token_num <= model_config_.max_step_token_num) {
    TestCommunicatePerformance({test_token_num, model_config_.hidden_units}, model_config_.weight_data_type);
    test_token_num *= 2;
  }
}

void CommunicationPerformanceRunner::TestCommunicatePerformance(const std::vector<size_t>& shape, DataType data_type) {
  static constexpr size_t warm_up_rounds = 10;
  static constexpr size_t rounds = 100;
  SetHiddenUnitMeta(DEFAULT_MULTI_BATCH_ID, shape, data_type);
  if (context_->IsChief()) {  // master node send tensors
    // Use DEFAULT_MULTI_BATCH_ID for performance testing
    InitHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    // warm up
    for (size_t i = 0; i < warm_up_rounds; ++i) {
      SendHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    }
    KLLM_LOG_INFO << fmt::format("Start run model performance of {} rounds", rounds);
    auto start = std::chrono::high_resolution_clock::now();
    // send tensors
    for (size_t i = 0; i < rounds; ++i) {
      SendHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    }
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    FreeHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    std::cout << fmt::format("Master elapsed time: {} seconds of {} rounds\n", elapsed.count(), rounds);
    std::cout << fmt::format("Tensor shape: {}, Tensor dtype: {} TP: {}\n", Vector2Str(shape), GetTypeString(data_type),
                             model_config_.tensor_para_size);
    std::cout << fmt::format("Average time per round: {} ms\n\n", (elapsed.count() / rounds) * 1000);

  } else {  // worker node receive tensors
    // warm up
    InitHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    for (size_t i = 0; i < warm_up_rounds; ++i) {
      RecvHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    }
    KLLM_LOG_INFO << fmt::format("Start Receive Tensors  of {} rounds", rounds);
    auto start = std::chrono::high_resolution_clock::now();
    // receive
    for (size_t i = 0; i < rounds; ++i) {
      RecvHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    }
    FreeHiddenUnits(DEFAULT_MULTI_BATCH_ID);
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;

    std::cout << fmt::format("Worker elapsed time: {} seconds of {} rounds\n", elapsed.count(), rounds);
    std::cout << fmt::format("Tensor shape: {}, Tensor dtype: {} TP: {}\n", Vector2Str(shape), GetTypeString(data_type),
                             model_config_.tensor_para_size);
    std::cout << fmt::format("Average time per round: {} ms\n\n", (elapsed.count() / rounds) * 1000);
  }
}
}  // namespace ksana_llm
