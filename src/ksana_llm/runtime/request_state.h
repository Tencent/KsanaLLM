/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

enum class RequestState {
  kWaiting,
  kRunning,
  kSwapped,
  kFinished,
  kTransfer,
};
