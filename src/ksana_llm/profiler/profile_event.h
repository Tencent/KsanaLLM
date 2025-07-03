/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace ksana_llm {

class ProfileEvent {
 public:
  explicit ProfileEvent(const std::string& profile_event_name, int rank = 0) { PushEvent(profile_event_name, rank); }
  ~ProfileEvent() { PopEvent(); }

 private:
  static void PushEvent(const std::string& profile_event_name, int rank = 0);
  static void PopEvent();
};

#define PROFILE_EVENT_SCOPE(var_name, event_str, ...) \
  ksana_llm::ProfileEvent __profile_event__##var_name((event_str), ##__VA_ARGS__)

}  // namespace ksana_llm
