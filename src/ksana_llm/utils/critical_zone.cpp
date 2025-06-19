/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/critical_zone.h"

#include <mutex>
namespace ksana_llm {

std::mutex g_device_computing_mtx;
void EnterDeviceComputingCriticalZone() { g_device_computing_mtx.lock(); }
void LeaveDeviceComputingCriticalZone() { g_device_computing_mtx.unlock(); }
}  // namespace ksana_llm
