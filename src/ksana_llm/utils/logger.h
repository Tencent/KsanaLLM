/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cassert>

#include "loguru.hpp"

namespace ksana_llm {

// Log level.
enum Level {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  FATAL = 4,
  ATTENTION = 5,
  COMMUNICATION = 6,
  MOE = 7,
  MODEL = 8,
  SCHEDULER = 9,
  MAIN = 10,
  MULTI_BATCH = 11
};

extern std::vector<std::string> g_detail_levels;


// Get log level from environment, this function called only once.
static std::vector<std::string> GetLogLevels() {
  const char* default_log_level = "INFO";
  const char* env_log_level = std::getenv("KLLM_LOG_LEVEL");
  std::string log_level_str = env_log_level ? env_log_level : default_log_level;

  // Split the input levels by comma (',') and store in a vector
  std::stringstream ss(log_level_str);
  std::vector<std::string> input_levels;
  std::string str;
  while (std::getline(ss, str, ',')) {
    input_levels.push_back(str);
  }

  const std::unordered_map<std::string, Level> log_name_to_level = {{"DEBUG", Level::DEBUG},
                                                                    {"INFO", Level::INFO},
                                                                    {"WARNING", Level::WARNING},
                                                                    {"ERROR", Level::ERROR},
                                                                    {"FATAL", Level::FATAL},
                                                                    {"ATTENTION", Level::ATTENTION},
                                                                    {"COMMUNICATION", Level::COMMUNICATION},
                                                                    {"MOE", Level::MOE},
                                                                    {"MODEL", Level::MODEL},
                                                                    {"MAIN", Level::MAIN},
                                                                    {"MULTI_BATCH", Level::MULTI_BATCH},
                                                                    {"SCHEDULER", Level::SCHEDULER}};
  std::vector<std::string> valid_levels;
  for (auto& lvl : input_levels) {
    auto it = log_name_to_level.find(lvl);
    if (it != log_name_to_level.end()) {
      valid_levels.push_back(it->first);
    } else {
      std::cerr << "Warning: Unkown log level " << lvl << ", skip this level." << std::endl;
    }
  }
  if (valid_levels.empty()) {
    valid_levels.push_back("INFO");
  }
  return valid_levels;
}

// Get log filename from environment, called once.
static std::string GetLogFile() {
  const char* default_log_file = "log/ksana_llm.log";
  const char* env_log_file = std::getenv("KLLM_LOG_FILE");
  return env_log_file ? env_log_file : default_log_file;
}

void details_log_handler(void* user_data, const loguru::Message& message);

// Init logrun instance.
// base log level: INFO, DEBUG, WARNING, ERROR, FATAL
// details log level: ATTENTION, COMMUNICATION, MOE, MODEL, SCHEDULER
//   if debug is set, all details category will be set
inline void InitLoguru(bool force = false) {
  const std::vector<std::string> input_log_levels = GetLogLevels();
  loguru::Verbosity verbosity = loguru::Verbosity_INVALID;
  // 1. check if have debug
  bool has_debug = std::any_of(input_log_levels.begin(), input_log_levels.end(), [](const std::string& str) {
    return str == "DEBUG";
  });
  if (has_debug) {
    verbosity = loguru::Verbosity_MAX;
  }

  // 2. check if have details category
  const std::vector<std::string> all_details_levels = {"ATTENTION", "COMMUNICATION", "MOE",        "MODEL",
                                                       "SCHEDULER", "MAIN",          "MULTI_BATCH"};
  for (const auto& level : input_log_levels) {
    bool has_details = std::any_of(all_details_levels.begin(), all_details_levels.end(),
                                   [&level](const std::string& str) { return level == str; });
    if (has_details) {
      verbosity = loguru::Verbosity_MAX;
      break;
    }
  }

  // 3. check if have base
  if (verbosity != loguru::Verbosity_MAX) {
    for (const auto& level : input_log_levels) {
      if (level == "INFO") {
        verbosity = loguru::Verbosity_INFO;
      } else if (level == "WARNING") {
        verbosity = loguru::Verbosity_WARNING;
      } else if (level == "ERROR") {
        verbosity = loguru::Verbosity_ERROR;
      } else if (level == "FATAL") {
        verbosity = loguru::Verbosity_FATAL;
      }
    }
  }

  loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
  static bool kIsLoggerInitialized = false;
  if (!kIsLoggerInitialized || force) {
    if (verbosity == loguru::Verbosity_MAX) {
      g_detail_levels.clear();
      if (has_debug) {
        g_detail_levels.push_back("DEBUG");
        for (auto& level : all_details_levels) {
          g_detail_levels.push_back(level);
        }
      } else {
        for (auto& level : input_log_levels) {
          g_detail_levels.push_back(level);
        }
      }
      loguru::add_file(GetLogFile().c_str(), loguru::Append, loguru::Verbosity_INFO);
      loguru::add_callback("CATEGORY", details_log_handler, nullptr, verbosity);
    } else {
      loguru::add_file(GetLogFile().c_str(), loguru::Append, verbosity);
    }
    kIsLoggerInitialized = true;
  }
}

#define NO_CC_IF if  // For CodeCC compatibility.

#define KLLM_LOG_DEBUG LOG_S(1) << "DEBUG| " << __FUNCTION__ << " | "
#define KLLM_LOG_ATTENTION LOG_S(1) << "ATTENTION| " << __FUNCTION__ << " | "
#define KLLM_LOG_COMMUNICATION LOG_S(1) << "COMMUNICATION| " << __FUNCTION__ << " | "
#define KLLM_LOG_MOE LOG_S(1) << "MOE| " << __FUNCTION__ << " | "
#define KLLM_LOG_MODEL LOG_S(1) << "MODEL| " << __FUNCTION__ << " | "
#define KLLM_LOG_SCHEDULER LOG_S(1) << "SCHEDULER| " << __FUNCTION__ << " | "
#define KLLM_LOG_MAIN LOG_S(1) << "MAIN| " << __FUNCTION__ << " | "
#define KLLM_LOG_MULTI_BATCH LOG_S(1) << "MULTI_BATCH| " << __FUNCTION__ << " | "

#define KLLM_LOG_INFO LOG_S(INFO)
#define KLLM_LOG_WARNING LOG_S(WARNING)
#define KLLM_LOG_ERROR LOG_S(ERROR)
#define KLLM_LOG_FATAL LOG_S(FATAL)

[[noreturn]] inline void ThrowRuntimeError(const char* const file, int const line, std::string const& info) {
  const std::string message = fmt::format("{} ({}:{})", info, file, line);
  KLLM_LOG_ERROR << message;
  throw std::runtime_error(message);
}

inline void CheckAssert(bool result, const char* const file, int const line, std::string const& info) {
  if (!result) {
    ThrowRuntimeError(file, line, info);
  }
}

// Get current time in sec.
inline uint64_t GetCurrentTime() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// Get current time in ms.
inline uint64_t GetCurrentTimeInMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

#define KLLM_CHECK(val) CheckAssert(val, __FILE__, __LINE__, "Assertion failed")
#define KLLM_CHECK_WITH_INFO(val, info)                                                           \
  do {                                                                                            \
    bool is_valid_val = (val);                                                                    \
    if (!is_valid_val) {                                                                          \
      CheckAssert(is_valid_val, __FILE__, __LINE__, fmt::format("Assertion failed: {}", (info))); \
    }                                                                                             \
  } while (0)

#define KLLM_THROW(info) ThrowRuntimeError(__FILE__, __LINE__, info)

}  // namespace ksana_llm
