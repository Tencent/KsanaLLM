/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"

namespace ksana_llm {

class ScheduleStrategyFactory {
 public:
  // Create a scheduler strategy.
  static std::shared_ptr<BaseScheduleStrategy> CreateScheduleStrategy(
      const BatchSchedulerConfig &batch_scheduler_config, int tp_num);
};

}  // namespace ksana_llm
