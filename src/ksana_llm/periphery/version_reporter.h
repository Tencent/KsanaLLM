/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <arpa/inet.h>
#include <curl/curl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {
  // Structure to hold version report options
  struct ReportOption {};
  
  // Structure to hold version information
  struct VersionInfo {};
  
  // Structure to hold the result of a report
  struct ReportResult {};

  class VersionReporter {
    public:
      static VersionReporter& GetInstance() {
        static VersionReporter instance;
        return instance;
      }

      bool Init(const ReportOption& option = ReportOption()) {return false;};

      bool Start() { return true; }

      void StopReporting() { return; }

      void Destroy() { return; }

      void Stop() { return; }
  };
}  // namespace ksana_llm
