/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "logger.h"

namespace ksana_llm {

std::vector<std::string> g_detail_levels;

void details_log_handler(void* user_data, const loguru::Message& message) {
  const std::string output_file = GetLogFile();

  for (const auto& log_level : g_detail_levels) {
    if (message.message && std::string_view(message.message).find(log_level) != std::string_view::npos) {
      std::ofstream out(output_file, std::ios::app);  // Open file in append mode
      if (out.is_open()) {
        out << message.preamble << message.message << std::endl;
        out.close();
      } else {
        std::cerr << "Error opening file for writing: " << output_file << std::endl;
      }
    }
  }
}

IntervalLogger::IntervalLogger(uint64_t interval_ms) : interval_ms_(interval_ms), last_time_ms_(GetCurrentTimeInMs()) {}

bool IntervalLogger::ShouldLog() {
  const uint64_t current_time_ms = GetCurrentTimeInMs();
  if (current_time_ms - last_time_ms_ < interval_ms_) {
    return false;
  }
  last_time_ms_ = current_time_ms;
  return true;
}

}  // namespace ksana_llm
