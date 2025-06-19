/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"

namespace ksana_llm {

void BaseScheduleStrategy::SetBatchState(std::shared_ptr<BatchState> batch_state) { batch_state_ = batch_state; }

void BaseScheduleStrategy::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) {
  cache_manager_ = cache_manager;
}

}  // namespace ksana_llm
