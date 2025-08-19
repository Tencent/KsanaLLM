/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"

namespace ksana_llm {

void BaseScheduleStrategy::SetBatchState(std::shared_ptr<BatchState> batch_state) { batch_state_ = batch_state; }

void BaseScheduleStrategy::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) {
  cache_manager_ = cache_manager;
}

void BaseScheduleStrategy::SetSharedCounter(std::shared_ptr<SchedulerSharedCounter> scheduler_shared_counter) {
  scheduler_shared_counter_ = scheduler_shared_counter;
}

void BaseScheduleStrategy::SetSchedulerTickTok(std::shared_ptr<SchedulerTickTok> scheduler_ticktok) {
  scheduler_ticktok_ = scheduler_ticktok;
}

void BaseScheduleStrategy::SetDataParaGroupId(size_t dp_group_id) { dp_group_id_ = dp_group_id; }

}  // namespace ksana_llm
