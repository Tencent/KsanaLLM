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

#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/environment.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/exporters/ostream/metric_exporter_factory.h"
#include "opentelemetry/exporters/ostream/span_exporter_factory.h"
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
#include "opentelemetry/sdk/trace/exporter.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/simple_processor_factory.h"
#include "opentelemetry/sdk/trace/tracer_context.h"
#include "opentelemetry/sdk/trace/tracer_context_factory.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/semantic_conventions.h"

namespace ksana_llm {
const char SCOPE_VERSION[] = "1.2.0";

class NullBuffer : public std::streambuf {
  virtual int overflow(int c) { return c; }
};
template <typename T>
class HttpTextMapCarrier : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  explicit HttpTextMapCarrier(T& headers) : headers_(headers) {}
  HttpTextMapCarrier() = default;
  virtual opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
    auto it = headers_.find(std::string(key));
    if (it != headers_.end()) {
      return it->second;
    }
    return "";
  }

  virtual void Set(opentelemetry::nostd::string_view key, opentelemetry::nostd::string_view value) noexcept override {}

  T headers_;
};

// Use to collect profile data from different modules, must be thread-safe.
class Profiler {
 public:
  struct Monitor {
    // Step 1: Define the metrics object, should consider using a Counter or a Histogram.
    struct Call {
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> forward_req_total_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> forward_req_error_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> forward_req_timeout_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> forward_req_aborted_num;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> forward_cost_time_ms;

      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> prefix_cache_hit_req_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> prefix_cache_hit_token_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> prefix_cache_hit_block_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> full_prompt_matched_req_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> full_prompt_matched_block_num;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> metric_zero_output_token_num;

      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> batch_scheduler_batch_size;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> batch_scheduler_waiting_size;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> batch_scheduler_swapped_size;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> batch_manager_schedule_ms;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> req_total_cost_in_queue_ms;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> token_num_in_batch;
      std::unique_ptr<opentelemetry::metrics::Histogram<double>> token_fill_ratio;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> block_num_free;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> block_num_used;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> batch_scheduler_pending_swapin_size;
      std::unique_ptr<opentelemetry::metrics::Counter<uint64_t>> batch_scheduler_pending_swapout_size;

      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> time_to_first_token_ms;
      std::unique_ptr<opentelemetry::metrics::Histogram<double>> time_to_per_output_token_ms;

      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> metric_input_tokens_num;
      std::unique_ptr<opentelemetry::metrics::Histogram<uint64_t>> metric_output_token_num;
    } call;
  };

  Profiler();
  void Init(const ProfilerConfig& profiler_config);
  ~Profiler();

  void InitTracer();

  void CleanupTracer();

  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> GetTracer(std::string tracer_name);

  void InitMetrics();

  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> GetMeter(std::string meter_name);

  void CleanupMetrics();

  Monitor monitor_;

 private:
  std::string trace_export_url_;
  std::string metrics_export_url_;
  uint64_t export_interval_millis_;
  uint64_t export_timeout_millis_;
  opentelemetry::sdk::common::AttributeMap attr_;

  // Step 2: Define the metrics key name.
  static constexpr std::string_view kForwardReqTotalNum = "forward_req_total_num";
  static constexpr std::string_view kForwardReqErrorNum = "forward_req_error_num";
  static constexpr std::string_view kForwardReqTimeoutNum = "forward_req_timeout_num";
  static constexpr std::string_view kForwardReqAbortedNum = "forward_req_aborted_num";
  static constexpr std::string_view kForwardCostTimeMs = "forward_cost_time_ms";

  static constexpr std::string_view kPrefixCacheHitReqNum = "prefix_cache_hit_req_num";
  static constexpr std::string_view kPrefixCacheHitTokenNum = "prefix_cache_hit_token_num";
  static constexpr std::string_view kPrefixCacheHitBlockNum = "prefix_cache_hit_block_num";
  static constexpr std::string_view kFullPromptMatchedReqNum = "full_prompt_matched_req_num";
  static constexpr std::string_view kFullPromptMatchedBlockNum = "full_prompt_matched_block_num";

  static constexpr std::string_view kBatchSchedulerBatchSize = "batch_scheduler_batch_size";
  static constexpr std::string_view kBatchSchedulerWaitingSize = "batch_scheduler_waiting_size";
  static constexpr std::string_view kBatchSchedulerSwappedSize = "batch_scheduler_swapped_size";
  static constexpr std::string_view kBatchManagerScheduleMs = "batch_manager_schedule_ms";
  static constexpr std::string_view kReqTotalCostInQueueMs = "req_total_cost_in_queue_ms";
  static constexpr std::string_view kTokenNumInBatch = "token_num_in_batch";
  static constexpr std::string_view kTokenFillRatio = "token_fill_ratio";
  static constexpr std::string_view kBlockNumFree = "block_num_free";
  static constexpr std::string_view kBlockNumUsed = "block_num_used";
  static constexpr std::string_view kBatchSchedulerPendingSwapInSize = "batch_scheduler_pending_swapin_size";
  static constexpr std::string_view kBatchSchedulerPendingSwapOutSize = "batch_scheduler_pending_swapout_size";

  static constexpr std::string_view kTimeToFirstTokenMs = "time_to_first_token_ms";
  static constexpr std::string_view kTimeToPerOutputTokenMs = "time_to_per_output_token_ms";

  static constexpr std::string_view kInputTokensNum = "metric_input_tokens_num";
  static constexpr std::string_view kOutputTokenNum = "metric_output_token_num";
  static constexpr std::string_view kZeroOutputTokenNum = "metric_zero_output_token_num";

  // Step 3: Add the metrics key name to the metrics_ set.
  std::unordered_set<std::string_view> metrics_ = {kForwardReqTotalNum,
                                                   kForwardReqErrorNum,
                                                   kForwardReqTimeoutNum,
                                                   kForwardReqAbortedNum,
                                                   kForwardCostTimeMs,
                                                   kPrefixCacheHitReqNum,
                                                   kPrefixCacheHitTokenNum,
                                                   kPrefixCacheHitBlockNum,
                                                   kFullPromptMatchedReqNum,
                                                   kFullPromptMatchedBlockNum,
                                                   kBatchSchedulerBatchSize,
                                                   kBatchSchedulerWaitingSize,
                                                   kBatchSchedulerSwappedSize,
                                                   kBatchManagerScheduleMs,
                                                   kReqTotalCostInQueueMs,
                                                   kTokenNumInBatch,
                                                   kTokenFillRatio,
                                                   kBlockNumFree,
                                                   kBlockNumUsed,
                                                   kBatchSchedulerPendingSwapInSize,
                                                   kBatchSchedulerPendingSwapOutSize,
                                                   kTimeToFirstTokenMs,
                                                   kTimeToPerOutputTokenMs,
                                                   kInputTokensNum,
                                                   kOutputTokenNum,
                                                   kZeroOutputTokenNum};
};
}  // namespace ksana_llm
