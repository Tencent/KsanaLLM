
/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/periphery/version_reporter.h"
#include <gtest/gtest.h>

namespace ksana_llm {

class VersionReporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    option.report_interval = 5000;       // 5秒
    option.fail_report_interval = 2000;  // 2秒
  }

  void TearDown() override { VersionReporter::GetInstance().Destroy(); }

  ReportOption option;
};

TEST_F(VersionReporterTest, Initialization) {
  bool init_result = VersionReporter::GetInstance().Init(option);
  ASSERT_TRUE(init_result);
  ASSERT_TRUE(VersionReporter::GetInstance().IsInitialized());
  VersionReporter::GetInstance().StopReporting();
  VersionReporter::GetInstance().Destroy();
}

TEST_F(VersionReporterTest, DefaultInitialization) {
  bool init_result = VersionReporter::GetInstance().Init();
  ASSERT_TRUE(init_result);
  ASSERT_TRUE(VersionReporter::GetInstance().IsInitialized());

  // 检查默认的 option 是否正确
  ReportOption default_option;
  ASSERT_EQ(VersionReporter::GetInstance().GetReportInterval(), default_option.report_interval);
  ASSERT_EQ(VersionReporter::GetInstance().GetFailReportInterval(), default_option.fail_report_interval);
  VersionReporter::GetInstance().StopReporting();
  VersionReporter::GetInstance().Destroy();
}

TEST_F(VersionReporterTest, ReportIntervals) {
  VersionReporter::GetInstance().Init(option);
  ASSERT_EQ(VersionReporter::GetInstance().GetReportInterval(), option.report_interval);
  ASSERT_EQ(VersionReporter::GetInstance().GetFailReportInterval(), option.fail_report_interval);
  VersionReporter::GetInstance().StopReporting();
  VersionReporter::GetInstance().Destroy();
}

TEST_F(VersionReporterTest, ReportFunctionExecution) {
  VersionReporter::GetInstance().Init(option);

  // 模拟一段时间，确保 ReportFunction 被调用
  std::this_thread::sleep_for(std::chrono::seconds(10));

  // 检查 ReportFunction 是否执行成功
  ASSERT_NE(VersionReporter::GetInstance().GetLastReportTime(), 0);
  VersionReporter::GetInstance().StopReporting();
  VersionReporter::GetInstance().Destroy();
}

TEST_F(VersionReporterTest, StopAndDestroy) {
  VersionReporter::GetInstance().Init(option);
  VersionReporter::GetInstance().StopReporting();
  VersionReporter::GetInstance().Destroy();
  ASSERT_FALSE(VersionReporter::GetInstance().IsInitialized());
}

}  // namespace ksana_llm
