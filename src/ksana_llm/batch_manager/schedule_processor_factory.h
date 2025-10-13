/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/batch_manager/schedule_processor_interface.h"
#include "ksana_llm/utils/config/schedule_config_parser.h"

namespace ksana_llm {

class ScheduleProcessorFactory {
 public:
  // 根据配置创建调度处理器
  static std::unique_ptr<ScheduleProcessorInterface> CreateProcessor(const RuntimeConfig& runtime_config);

 private:
  ScheduleProcessorFactory() = default;
};

}  // namespace ksana_llm