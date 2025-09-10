/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "profiler.h"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

namespace ksana_llm {

// 定义一个派生自 ::testing::Test 的测试夹具类
class ProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 在每个测试之前调用，通过单例获取 Profiler 实例
    profiler = Singleton<Profiler>::GetInstance().get();

    // 准备测试用的配置
    test_config.metrics_export_url = "debug";
    test_config.resource_attributes = {{"service.name", "test_service"}, {"environment", "test"}, {"version", "1.0.0"}};
    test_config.export_interval_millis = 5000;
    test_config.export_timeout_millis = 2000;
  }

  void TearDown() override {
    // 清理单例实例，为下一个测试做准备
    Singleton<Profiler>::DeleteInstance();
    profiler = nullptr;
  }

  Profiler* profiler = nullptr;
  ProfilerConfig test_config;
};

// 测试默认构造函数和初始值
TEST_F(ProfilerTest, DefaultConstructorInitialization) {
  // 创建新的实例来测试默认构造
  auto new_profiler = std::make_unique<Profiler>();

  // 验证默认值（这些值应该通过检查私有成员或公共行为来验证）
  // 由于成员是私有的，我们通过行为来测试
  EXPECT_NO_THROW(new_profiler->ReportMetric("test_metric", 1.0));
  EXPECT_NO_THROW(new_profiler->ReportCounter("test_counter", 1));
}

// 测试拷贝和移动构造函数被删除
TEST_F(ProfilerTest, CopyMoveConstructorsDeleted) {
  // 这个测试在编译时验证，如果能编译通过说明删除成功
  static_assert(!std::is_copy_constructible_v<Profiler>);
  static_assert(!std::is_copy_assignable_v<Profiler>);
  static_assert(!std::is_move_constructible_v<Profiler>);
  static_assert(!std::is_move_assignable_v<Profiler>);
}

// 测试 InitMetrics 方法 - HTTP 导出器
TEST_F(ProfilerTest, InitMetricsWithHttpExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "http://localhost:4318/v1/metrics";

  EXPECT_NO_THROW(profiler->InitMetrics(config));

  // 测试初始化后可以正常报告指标
  EXPECT_NO_THROW(profiler->ReportMetric("http_test_metric", 42.5));
  EXPECT_NO_THROW(profiler->ReportCounter("http_test_counter", 10));
}

// 测试 InitMetrics 方法 - Debug 导出器
TEST_F(ProfilerTest, InitMetricsWithDebugExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "debug";

  EXPECT_NO_THROW(profiler->InitMetrics(config));

  // 测试初始化后可以正常报告指标
  EXPECT_NO_THROW(profiler->ReportMetric("debug_test_metric", 123.45));
  EXPECT_NO_THROW(profiler->ReportCounter("debug_test_counter", 5));
}

// 测试 InitMetrics 方法 - DEBUG 大写导出器
TEST_F(ProfilerTest, InitMetricsWithDebugUppercaseExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "DEBUG";

  EXPECT_NO_THROW(profiler->InitMetrics(config));

  // 测试初始化后可以正常报告指标
  EXPECT_NO_THROW(profiler->ReportMetric("debug_upper_metric", 67.89));
  EXPECT_NO_THROW(profiler->ReportCounter("debug_upper_counter", 15));
}

// 测试 InitMetrics 方法 - 默认导出器（黑洞）
TEST_F(ProfilerTest, InitMetricsWithDefaultExporter) {
  ProfilerConfig config = test_config;
  config.metrics_export_url = "unknown_protocol";

  EXPECT_NO_THROW(profiler->InitMetrics(config));

  // 测试初始化后可以正常报告指标
  EXPECT_NO_THROW(profiler->ReportMetric("default_test_metric", 99.99));
  EXPECT_NO_THROW(profiler->ReportCounter("default_test_counter", 25));
}

// 测试 InitMetrics 方法 - 空配置
TEST_F(ProfilerTest, InitMetricsWithEmptyConfig) {
  ProfilerConfig empty_config;

  EXPECT_NO_THROW(profiler->InitMetrics(empty_config));

  // 即使配置为空，也应该能正常工作
  EXPECT_NO_THROW(profiler->ReportMetric("empty_config_metric", 1.1));
  EXPECT_NO_THROW(profiler->ReportCounter("empty_config_counter", 1));
}

// 测试 InitMetrics 方法 - 包含资源属性
TEST_F(ProfilerTest, InitMetricsWithResourceAttributes) {
  EXPECT_NO_THROW(profiler->InitMetrics(test_config));

  // 验证可以报告指标
  EXPECT_NO_THROW(profiler->ReportMetric("attr_test_metric", 50.0));
  EXPECT_NO_THROW(profiler->ReportCounter("attr_test_counter", 20));
}

// 测试 ReportMetric 方法 - 新指标
TEST_F(ProfilerTest, ReportMetricNewMetric) {
  profiler->InitMetrics(test_config);

  // 第一次报告新指标
  EXPECT_NO_THROW(profiler->ReportMetric("new_metric_test", 123.456));

  // 再次报告相同指标（应该重用已创建的）
  EXPECT_NO_THROW(profiler->ReportMetric("new_metric_test", 789.012));
}

// 测试 ReportMetric 方法 - 现有指标
TEST_F(ProfilerTest, ReportMetricExistingMetric) {
  profiler->InitMetrics(test_config);

  // 创建指标
  profiler->ReportMetric("existing_metric", 1.0);

  // 多次报告相同指标
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 2.0));
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 3.0));
  EXPECT_NO_THROW(profiler->ReportMetric("existing_metric", 4.0));
}

// 测试 ReportMetric 方法 - 各种数值
TEST_F(ProfilerTest, ReportMetricVariousValues) {
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportMetric("zero_metric", 0.0));
  EXPECT_NO_THROW(profiler->ReportMetric("negative_metric", -123.45));
  EXPECT_NO_THROW(profiler->ReportMetric("large_metric", 1e10));
  EXPECT_NO_THROW(profiler->ReportMetric("small_metric", 1e-10));
  EXPECT_NO_THROW(profiler->ReportMetric("inf_metric", std::numeric_limits<double>::infinity()));
}

// 测试 ReportCounter 方法 - 新计数器
TEST_F(ProfilerTest, ReportCounterNewCounter) {
  profiler->InitMetrics(test_config);

  // 第一次报告新计数器
  EXPECT_NO_THROW(profiler->ReportCounter("new_counter_test", 100));

  // 再次报告相同计数器（应该重用已创建的）
  EXPECT_NO_THROW(profiler->ReportCounter("new_counter_test", 200));
}

// 测试 ReportCounter 方法 - 现有计数器
TEST_F(ProfilerTest, ReportCounterExistingCounter) {
  profiler->InitMetrics(test_config);

  // 创建计数器
  profiler->ReportCounter("existing_counter", 1);

  // 多次报告相同计数器
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 2));
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 3));
  EXPECT_NO_THROW(profiler->ReportCounter("existing_counter", 4));
}

// 测试 ReportCounter 方法 - 各种数值
TEST_F(ProfilerTest, ReportCounterVariousValues) {
  profiler->InitMetrics(test_config);

  EXPECT_NO_THROW(profiler->ReportCounter("zero_counter", 0));
  EXPECT_NO_THROW(profiler->ReportCounter("small_counter", 1));
  EXPECT_NO_THROW(profiler->ReportCounter("large_counter", 1000000));
  EXPECT_NO_THROW(profiler->ReportCounter("max_counter", std::numeric_limits<int64_t>::max()));
}

// 测试在未初始化时报告指标
TEST_F(ProfilerTest, ReportMetricsWithoutInitialization) {
  // 不调用 InitMetrics，直接报告指标
  // 这应该不会崩溃，但可能不会正常工作
  EXPECT_NO_THROW(profiler->ReportMetric("uninit_metric", 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter("uninit_counter", 1));
}

// 测试析构函数
TEST_F(ProfilerTest, Destructor) {
  // 创建一个临时的 Profiler 实例来测试析构
  {
    auto temp_profiler = std::make_unique<Profiler>();
    temp_profiler->InitMetrics(test_config);
    // temp_profiler 在作用域结束时自动析构
  }

  // 如果能执行到这里，说明析构函数正常工作
  SUCCEED();
}

// 测试单例模式
TEST_F(ProfilerTest, SingletonPattern) {
  auto instance1 = Singleton<Profiler>::GetInstance();
  auto instance2 = Singleton<Profiler>::GetInstance();

  // 两次获取应该返回同一个实例
  EXPECT_EQ(instance1.get(), instance2.get());
  EXPECT_EQ(instance1.get(), profiler);
}

// 测试线程安全性
TEST_F(ProfilerTest, ThreadSafety) {
  profiler->InitMetrics(test_config);

  const int num_threads = 10;
  const int reports_per_thread = 100;
  std::vector<std::thread> threads;

  // 启动多个线程同时报告指标
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, i, reports_per_thread]() {
      for (int j = 0; j < reports_per_thread; ++j) {
        profiler->ReportMetric("thread_metric_" + std::to_string(i), j * 1.0);
        profiler->ReportCounter("thread_counter_" + std::to_string(i), j);
      }
    });
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 如果能执行到这里，说明线程安全测试通过
  SUCCEED();
}

// 测试大量指标创建
TEST_F(ProfilerTest, ManyMetrics) {
  profiler->InitMetrics(test_config);

  const int num_metrics = 1000;

  // 创建大量不同的指标
  for (int i = 0; i < num_metrics; ++i) {
    EXPECT_NO_THROW(profiler->ReportMetric("metric_" + std::to_string(i), i * 1.0));
    EXPECT_NO_THROW(profiler->ReportCounter("counter_" + std::to_string(i), i));
  }
}

// 测试指标名称边界情况
TEST_F(ProfilerTest, MetricNameEdgeCases) {
  profiler->InitMetrics(test_config);

  // 空字符串名称
  EXPECT_NO_THROW(profiler->ReportMetric("", 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter("", 1));

  // 很长的名称
  std::string long_name(1000, 'a');
  EXPECT_NO_THROW(profiler->ReportMetric(long_name, 1.0));
  EXPECT_NO_THROW(profiler->ReportCounter(long_name, 1));

  // 包含特殊字符的名称
  EXPECT_NO_THROW(profiler->ReportMetric("metric.with.dots", 1.0));
  EXPECT_NO_THROW(profiler->ReportMetric("metric_with_underscores", 1.0));
  EXPECT_NO_THROW(profiler->ReportMetric("metric-with-dashes", 1.0));
}

}  // namespace ksana_llm