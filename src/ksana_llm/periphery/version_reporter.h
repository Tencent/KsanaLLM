
/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

struct ReportOption {};

namespace ksana_llm {
class VersionReporter {
 public:
  // Singleton instance getter
  static VersionReporter& GetInstance() {
    static VersionReporter instance;
    return instance;
  }
  // Initialize version reporting with given options
  void Init(const ReportOption& option = ReportOption());

  // Stop the version reporting
  void StopReporting();

  // Destroy the version reporting instance
  void Destroy();
};
}  //  namespace ksana_llm
