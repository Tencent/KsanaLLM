/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include "loguru.hpp"

namespace ksana_llm {

// Log level.
enum Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

// Get log level from environment, this function called only once.
static Level GetLogLevel() {
  const char* default_log_level = "INFO";
  const char* env_log_level = std::getenv("KLLM_LOG_LEVEL");
  std::string log_level_str = env_log_level ? env_log_level : default_log_level;

  std::unordered_map<std::string, Level> log_name_to_level = {
      {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"ERROR", ERROR}, {"FATAL", FATAL}};

  Level level = Level::INFO;
  if (log_name_to_level.find(log_level_str) != log_name_to_level.end()) {
    level = log_name_to_level[log_level_str];
  }
  return level;
}

// Get log filename from environment, called once.
static std::string GetLogFile() {
  const char* default_log_file = "log/ksana_llm.log";
  const char* env_log_file = std::getenv("KLLM_LOG_FILE");
  return env_log_file ? env_log_file : default_log_file;
}

// Get name from log level.
inline std::string GetLevelName(const Level level) {
  switch (level) {
    case DEBUG:
      return "DEBUG";
    case INFO:
      return "INFO";
    case WARNING:
      return "WARNING";
    case ERROR:
      return "ERROR";
    case FATAL:
      return "FATAL";
    default:
      return "Invalid: " + std::to_string(level);
  }
}

// Init logrun instance.
inline void InitLoguru() {
  Level log_level = GetLogLevel();

  loguru::Verbosity verbosity = loguru::Verbosity_MAX;
  if (log_level <= Level::DEBUG) {
    verbosity = loguru::Verbosity_MAX;
  } else if (log_level == Level::INFO) {
    verbosity = loguru::Verbosity_INFO;
  } else if (log_level == Level::WARNING) {
    verbosity = loguru::Verbosity_WARNING;
  } else if (log_level == Level::ERROR) {
    verbosity = loguru::Verbosity_ERROR;
  } else if (log_level == Level::FATAL) {
    verbosity = loguru::Verbosity_FATAL;
  }

  loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
  static bool kIsLoggerInitialized = false;
  if (!kIsLoggerInitialized) {
    loguru::add_file(GetLogFile().c_str(), loguru::Append, verbosity);
    kIsLoggerInitialized = true;
  }
}

#define NO_CC_IF if  // For CodeCC compatibility.

#define KLLM_LOG_DEBUG LOG_S(1) << __FUNCTION__ << "| "
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
