/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/critical_zone.h"

#include <mutex>

#include "ksana_llm/profiler/profile_event.h"

namespace ksana_llm {

std::mutex g_device_computing_mtx;
void EnterDeviceComputingCriticalZone() {
  PROFILE_EVENT_SCOPE(EnterDeviceComputingCriticalZone, "EnterDeviceComputingCriticalZone");
  g_device_computing_mtx.lock();
}

void LeaveDeviceComputingCriticalZone() {
  PROFILE_EVENT_SCOPE(LeaveDeviceComputingCriticalZone, "LeaveDeviceComputingCriticalZone");
  g_device_computing_mtx.unlock();
}
}  // namespace ksana_llm
