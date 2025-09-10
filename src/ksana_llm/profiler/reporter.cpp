/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/reporter.h"

namespace ksana_llm {

TimeReporter::TimeReporter(const std::string &name, TimeUnit time_unit) {
  name_ = name;
  time_unit_ = time_unit;
  start_ = GetCurrentByUnit(time_unit_);
}

TimeReporter::~TimeReporter() {}

inline time_t TimeReporter::GetCurrentByUnit(TimeUnit time_unit) {
  switch (time_unit) {
    case TimeUnit::TIME_MS:
      return ProfileTimer::GetCurrentTimeInMs();
      break;
    case TimeUnit::TIME_US:
      return ProfileTimer::GetCurrentTimeInUs();
      break;
    case TimeUnit::TIME_NS:
      return ProfileTimer::GetCurrentTimeInNs();
    default:
      KLLM_CHECK_WITH_INFO(false, "time unit is not supported.");
      break;
  }
  return 0;
}
}  // namespace ksana_llm
