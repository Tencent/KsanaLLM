/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/strategy_factory.h"

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"

namespace ksana_llm {

std::shared_ptr<BaseScheduleStrategy> ScheduleStrategyFactory::CreateScheduleStrategy(
    const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config) {
  if (batch_scheduler_config.schedule_strategy == ScheduleStrategy::CONTINUOUS_BATCHING) {
    KLLM_LOG_DEBUG << "Continuous-batching scheduler created.";
    return std::make_shared<ContinuousBatchingStrategy>(batch_scheduler_config, runtime_config);
  }
  return nullptr;
}

}  // namespace ksana_llm
