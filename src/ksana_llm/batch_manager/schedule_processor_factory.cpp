/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/schedule_processor_factory.h"
#include "ksana_llm/batch_manager/async_schedule_processor.h"
#include "ksana_llm/batch_manager/schedule_processor.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

std::unique_ptr<ScheduleProcessorInterface> ScheduleProcessorFactory::CreateProcessor(
    const RuntimeConfig& runtime_config) {
  if (runtime_config.enable_async) {
    KLLM_LOG_INFO << "Creating AsyncScheduleProcessor (async mode enabled)";
    return std::make_unique<AsyncScheduleProcessor>();
  } else {
    KLLM_LOG_INFO << "Creating ScheduleProcessor (sync mode)";
    return std::make_unique<ScheduleProcessor>();
  }
}

}  // namespace ksana_llm