
/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ksana_llm {

// Structure to hold version report options
struct ReportOption {
  uint64_t report_interval = 24 * 60 * 60 * 1000;  // Interval for regular reports in milliseconds
  uint64_t fail_report_interval = 60 * 60 * 1000;  // Interval for failed reports in milliseconds
  std::string report_api = "/version/stat/add";    // API endpoint for reporting
  std::string report_host = "stat.ksana.woa.com";  // Host for reporting
};

// Structure to hold version information
struct VersionInfo {
  std::string app;             // Application name
  std::string server;          // Server name
  std::string ip;              // IP address
  std::string container_name;  // Container name
  std::string version;         // Version string
  std::string commit_hash;     // Commit hash
  std::string branch;          // Branch name
};

// Structure to hold the result of a report
struct ReportResult {
  bool is_succ = false;           // Indicates if the report was successful
  uint64_t last_report_time = 0;  // Timestamp of the last report
};

// Class for managing version reporting
class VersionReporter {
 public:
  // Singleton instance getter
  static VersionReporter& GetInstance() {
    static VersionReporter instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  VersionReporter(const VersionReporter&) = delete;
  VersionReporter& operator=(const VersionReporter&) = delete;

  // Initialize version reporting with given options
  bool Init(const ReportOption& option = ReportOption());

  // Stop the version reporting
  void StopReporting();

  // Destroy the version reporting instance
  void Destroy();

  // Check if the version reporting is initialized
  bool IsInitialized() const { return initialized_; }

  // Get the report interval
  uint64_t GetReportInterval() const { return option_ ? option_->report_interval : 0; }

  // Get the fail report interval
  uint64_t GetFailReportInterval() const { return option_ ? option_->fail_report_interval : 0; }

  // Get the last report time
  uint64_t GetLastReportTime() const { return report_result_.last_report_time; }

 private:
  VersionReporter() = default;
  ~VersionReporter() { Destroy(); }

  bool Start();
  void Stop();
  void ReportFunction();
  std::uint64_t ExecutionInterval();

  uint64_t task_id_{0};                                  // Task ID for reporting
  ReportResult report_result_;                           // Result of the last report
  std::shared_ptr<VersionInfo> version_info_ = nullptr;  // Pointer to version information
  std::shared_ptr<ReportOption> option_ = nullptr;       // Pointer to report options
  bool initialized_ = false;                             // Flag to indicate if initialized
  std::mutex version_report_mutex;                       // Mutex for thread safety
  bool version_report_init_flag_{false};                 // Flag to indicate if initialization is done
  std::thread report_thread_;
  std::atomic<bool> running_{false};
  std::mutex stop_cv_mutex_;
  std::condition_variable stop_cv_;
};

}  // namespace ksana_llm
