/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace ksana_llm {

class ProfileEvent {
 public:
  static void PushEvent(const std::string& profile_event_name, int rank = 0);
  static void PopEvent();
};

}  // namespace ksana_llm
