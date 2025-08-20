/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <thread>
#include <cstdlib>
#include "loguru.hpp"
#include "logger.h"
#include "test.h"

class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("KLLM_LOG_FILE", "test_log_file.log", 1);
  }

  void TearDown() override {
    std::remove("test_log_file.log");
    unsetenv("KLLM_LOG_LEVEL");
  }
};

TEST_F(LoggerTest, TestLoggingAttentionLevels) {
  setenv("KLLM_LOG_LEVEL", "ATTENTION", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_ATTENTION << "This is an ATTENTION level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("ATTENTION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an ATTENTION level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingCOMMUNICATIONLevels) {
  setenv("KLLM_LOG_LEVEL", "COMMUNICATION", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_COMMUNICATION << "This is an COMMUNICATION level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("COMMUNICATION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an COMMUNICATION level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingMOELevels) {
  setenv("KLLM_LOG_LEVEL", "MOE", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_MOE << "This is a MOE level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("MOE") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is a MOE level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingMODELLevels) {
  setenv("KLLM_LOG_LEVEL", "MODEL", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_MODEL << "This is a MODEL level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("MODEL") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is a MODEL level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingSCHEDULERLevel) {
  setenv("KLLM_LOG_LEVEL", "SCHEDULER", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_SCHEDULER << "This is an SCHEDULER level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("SCHEDULER") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an SCHEDULER level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingINFOLevel) {
  setenv("KLLM_LOG_LEVEL", "INFO", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_INFO << "This is an INFO level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("INFO") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an INFO level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingMAINLevel) {
  setenv("KLLM_LOG_LEVEL", "MAIN", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_MAIN << "This is an MAIN level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("MAIN") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an MAIN level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingDEBUGLevel) {
  setenv("KLLM_LOG_LEVEL", "DEBUG", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_DEBUG << "This is a DEBUG level log";
  KLLM_LOG_INFO << "This is an INFO level log";

  KLLM_LOG_MOE << "This is a MOE level log";
  KLLM_LOG_MODEL << "This is a MODEL level log";
  KLLM_LOG_SCHEDULER << "This is an SCHEDULER level log";
  KLLM_LOG_ATTENTION << "This is an ATTENTION level log";
  KLLM_LOG_COMMUNICATION << "This is an COMMUNICATION level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("DEBUG") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is a DEBUG level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("INFO") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an INFO level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("MOE") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is a MOE level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("MODEL") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is a MODEL level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("SCHEDULER") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an SCHEDULER level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("ATTENTION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an ATTENTION level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("COMMUNICATION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an COMMUNICATION level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingMultiLevels) {
  setenv("KLLM_LOG_LEVEL", "ATTENTION,COMMUNICATION", 1);
  ksana_llm::InitLoguru(true);
  KLLM_LOG_ATTENTION << "This is an ATTENTION level log";
  KLLM_LOG_COMMUNICATION << "This is an COMMUNICATION level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("ATTENTION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an ATTENTION level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("COMMUNICATION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an COMMUNICATION level log") != std::string::npos);
}

TEST_F(LoggerTest, TestLoggingLogLevels) {
  setenv("KLLM_LOG_LEVEL", "INFO,ATTENTION,COMMUNICATION", 1);
  ksana_llm::InitLoguru(true);
  auto levels = ksana_llm::GetLogLevels();
  EXPECT_EQ(levels.size(), 3);
  EXPECT_EQ(levels[0], "INFO");
  EXPECT_EQ(levels[1], "ATTENTION");
  EXPECT_EQ(levels[2], "COMMUNICATION");

  KLLM_LOG_INFO << "This is an INFO level log";
  KLLM_LOG_ATTENTION << "This is an ATTENTION level log";
  KLLM_LOG_COMMUNICATION << "This is an COMMUNICATION level log";

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("test_log_file.log");
  std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(log_contents.find("INFO") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an INFO level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("ATTENTION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an ATTENTION level log") != std::string::npos);
  EXPECT_TRUE(log_contents.find("COMMUNICATION") != std::string::npos);
  EXPECT_TRUE(log_contents.find("This is an COMMUNICATION level log") != std::string::npos);
}
