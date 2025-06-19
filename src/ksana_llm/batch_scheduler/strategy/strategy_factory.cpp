/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/strategy_factory.h"

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"

namespace ksana_llm {

std::shared_ptr<BaseScheduleStrategy> ScheduleStrategyFactory::CreateScheduleStrategy(
    const BatchSchedulerConfig &batch_scheduler_config, int tp_num) {
  if (batch_scheduler_config.schedule_strategy == ScheduleStrategy::CONTINUOUS_BATCHING) {
    KLLM_LOG_DEBUG << "Continuous-batching scheduler created.";
    return std::make_shared<ContinuousBatchingStrategy>(batch_scheduler_config, tp_num);
  }
  return nullptr;
}

}  // namespace ksana_llm
