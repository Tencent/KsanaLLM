/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "profiler.h"
#include <gtest/gtest.h>

namespace ksana_llm {

// 定义一个派生自 ::testing::Test 的测试夹具类
class ProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 在每个测试之前调用，初始化 Profiler 实例
    profiler = new Profiler();
  }

  void TearDown() override {
    // 在每个测试之后调用，清理资源
    delete profiler;
  }

  Profiler* profiler;
};

// 测试默认构造函数和初始值
TEST_F(ProfilerTest, DefaultInitialization) {
  EXPECT_EQ(profiler->export_interval_millis_, 60000);
  EXPECT_EQ(profiler->export_timeout_millis_, 30000);
  EXPECT_TRUE(profiler->trace_export_url_.empty());
  EXPECT_TRUE(profiler->metrics_export_url_.empty());
}

// 一个比较 AttributeMap 和 std::unordered_map<std::string, std::string> 的函数
bool CompareAttributeMaps(const opentelemetry::sdk::common::AttributeMap& attr_map,
                          const std::unordered_map<std::string, std::string>& std_map) {
  if (attr_map.size() != std_map.size()) {
    return false;
  }

  for (const auto& kv : attr_map) {
    auto it = std_map.find(kv.first);
    if (it == std_map.end()) {
      return false;
    }
    // 检查 AttributeValue 是否持有 std::string 类型的值
    if (std::holds_alternative<std::string>(kv.second)) {
      const std::string& attr_value = std::get<std::string>(kv.second);
      if (attr_value != it->second) {
        return false;
      }
    } else {
      // 如果不是 std::string 类型，无法比较，返回 false 或者根据需求处理
      return false;
    }
  }
  return true;
}

// 测试 Init 方法
TEST_F(ProfilerTest, InitMethod) {
  ProfilerConfig config;
  config.trace_export_url = "http://localhost:4318/v1/traces";
  config.metrics_export_url = "http://localhost:4318/v1/metrics";
  config.resource_attributes = {{"service.name", "test_service"}, {"environment", "test"}};
  config.export_interval_millis = 10000;
  config.export_timeout_millis = 5000;

  profiler->Init(config);

  EXPECT_EQ(profiler->trace_export_url_, config.trace_export_url);
  EXPECT_EQ(profiler->metrics_export_url_, config.metrics_export_url);
  EXPECT_EQ(profiler->export_interval_millis_, config.export_interval_millis);
  EXPECT_EQ(profiler->export_timeout_millis_, config.export_timeout_millis);
  EXPECT_TRUE(CompareAttributeMaps(profiler->attr_, config.resource_attributes));

  // 检查是否正确初始化了 Tracer
  auto tracer = profiler->GetTracer("test_tracer");
  EXPECT_NE(tracer.get(), nullptr);
}

// 测试 GetTracer 方法
TEST_F(ProfilerTest, GetTracer) {
  auto tracer = profiler->GetTracer("test_tracer");
  EXPECT_NE(tracer.get(), nullptr);
}

// 测试 GetMeter 方法
TEST_F(ProfilerTest, GetMeter) {
  auto meter = profiler->GetMeter("test_meter");
  EXPECT_NE(meter.get(), nullptr);
}

// 测试 CleanupTracer 方法
TEST_F(ProfilerTest, CleanupTracer) {
  profiler->CleanupTracer();
  // 清理后，TracerProvider 应该为空
  auto provider = opentelemetry::trace::Provider::GetTracerProvider();
  EXPECT_EQ(provider.get(), nullptr);
}

// 测试 CleanupMetrics 方法
TEST_F(ProfilerTest, CleanupMetrics) {
  profiler->CleanupMetrics();
  // 清理后，MeterProvider 应该为空
  auto provider = opentelemetry::metrics::Provider::GetMeterProvider();
  EXPECT_EQ(provider.get(), nullptr);
}

}  // namespace ksana_llm