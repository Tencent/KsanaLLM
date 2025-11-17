/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <streambuf>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>

#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/exporters/ostream/metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h"
#include "opentelemetry/ext/http/client/http_client.h"
#include "opentelemetry/metrics/provider.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/sdk/common/attribute_utils.h"
#include "opentelemetry/sdk/common/global_log_handler.h"
#include "opentelemetry/sdk/metrics/aggregation/default_aggregation.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h"
#include "opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h"
#include "opentelemetry/sdk/metrics/meter.h"
#include "opentelemetry/sdk/metrics/meter_context_factory.h"
#include "opentelemetry/sdk/metrics/meter_provider.h"
#include "opentelemetry/sdk/metrics/meter_provider_factory.h"
#include "opentelemetry/sdk/metrics/view/view_registry_factory.h"

namespace ksana_llm {
const char SCOPE_VERSION[] = "1.2.0";

class NullBuffer : public std::streambuf {
  virtual int overflow(int c) { return c; }
};

// Use to collect profile data from different modules, must be thread-safe.
class Profiler {
 public:
  // Constructor must be public because Singleton uses std::make_shared internally.
  // Do NOT construct directly outside of Singleton<Profiler>::GetInstance().
  Profiler();
  ~Profiler();

  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;
  Profiler(Profiler&&) = delete;
  Profiler& operator=(Profiler&&) = delete;

  void InitMetrics(const ProfilerConfig& profiler_config);
  void ReportMetric(const std::string& name, double value);
  void ReportCounter(const std::string& name, int64_t value);

 private:
  // Members
  std::string metrics_export_url_;
  bool is_initialized_ = false;
  uint64_t export_interval_millis_ = 60000;
  uint64_t export_timeout_millis_ = 30000;
  opentelemetry::sdk::common::AttributeMap resource_attributes_;
  std::unordered_map<std::string, std::string> data_attributes_;
  opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes_view_;
  opentelemetry::v1::nostd::shared_ptr<opentelemetry::v1::metrics::Meter> meter_;

  // PIMPL to avoid exposing TBB in public headers
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace ksana_llm
