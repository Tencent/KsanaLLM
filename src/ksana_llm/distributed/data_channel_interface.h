/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The pulibc interface for all data channels.
class DataChannelInterface {
 public:
  virtual ~DataChannelInterface() {}

  // For master node only.
  virtual Status Listen() = 0;

  // Close open port.
  virtual Status Close() = 0;

  // For normal node only.
  virtual Status Connect() = 0;

  // disconnect from master.
  virtual Status Disconnect() = 0;
};

}  // namespace ksana_llm
