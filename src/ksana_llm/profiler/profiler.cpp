/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ksana_llm/profiler/profiler.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

Profiler::Profiler() {
  // By default, data is export to the black hole file
  export_interval_millis_ = 60000;
  export_timeout_millis_ = 30000;
  InitTracer();
  InitMetrics();
}

void Profiler::Init(const ProfilerConfig& profiler_config) {
  trace_export_url_ = profiler_config.trace_export_url;
  metrics_export_url_ = profiler_config.metrics_export_url;
  for (const auto& kv : profiler_config.resource_attributes) {
    attr_[kv.first] = kv.second;
  }
  export_interval_millis_ = profiler_config.export_interval_millis;
  export_timeout_millis_ = profiler_config.export_timeout_millis;
  InitTracer();
  InitMetrics();
}

Profiler::~Profiler() {
  CleanupTracer();
  CleanupMetrics();
}

void Profiler::InitTracer() {
  std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter;

  if (trace_export_url_.substr(0, 4) == "http") {
    opentelemetry::exporter::otlp::OtlpHttpExporterOptions exporter_options;
    exporter_options.url = trace_export_url_;
    exporter = opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create(exporter_options);
  } else if (trace_export_url_ == "debug" || trace_export_url_ == "DEBUG") {
    exporter = opentelemetry::exporter::trace::OStreamSpanExporterFactory::Create(std::cout);
  } else {
    // By default, data is export to the black hole file
    exporter = opentelemetry::exporter::trace::OStreamSpanExporterFactory::Create(null_stream);
  }

  auto processor = opentelemetry::sdk::trace::SimpleSpanProcessorFactory::Create(std::move(exporter));
  std::vector<std::unique_ptr<opentelemetry::sdk::trace::SpanProcessor>> processors;
  processors.push_back(std::move(processor));

  // Default is an always-on sampler.
  std::unique_ptr<opentelemetry::sdk::trace::TracerContext> context =
      opentelemetry::sdk::trace::TracerContextFactory::Create(std::move(processors),
                                                              opentelemetry::sdk::resource::Resource::Create(attr_));
  std::shared_ptr<opentelemetry::trace::TracerProvider> provider =
      opentelemetry::sdk::trace::TracerProviderFactory::Create(std::move(context));

  // Set the global trace provider
  opentelemetry::trace::Provider::SetTracerProvider(provider);

  // set global propagator
  opentelemetry::context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
      opentelemetry::nostd::shared_ptr<opentelemetry::context::propagation::TextMapPropagator>(
          new opentelemetry::trace::propagation::HttpTraceContext()));
}

void Profiler::CleanupTracer() {
  std::shared_ptr<opentelemetry::trace::TracerProvider> none;
  opentelemetry::trace::Provider::SetTracerProvider(none);
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> Profiler::GetTracer(std::string tracer_name) {
  auto provider = opentelemetry::trace::Provider::GetTracerProvider();
  return provider->GetTracer(tracer_name);
}

void Profiler::InitMetrics() {
  std::unique_ptr<opentelemetry::sdk::metrics::PushMetricExporter> exporter;
  if (metrics_export_url_.substr(0, 4) == "http") {
    opentelemetry::exporter::otlp::OtlpHttpMetricExporterOptions exporter_options;
    exporter_options.url = metrics_export_url_;
    exporter_options.aggregation_temporality = opentelemetry::exporter::otlp::PreferredAggregationTemporality::kDelta;
    exporter_options.content_type = opentelemetry::exporter::otlp::HttpRequestContentType::kJson;
    exporter = opentelemetry::exporter::otlp::OtlpHttpMetricExporterFactory::Create(exporter_options);
  } else if (metrics_export_url_ == "debug" || metrics_export_url_ == "DEBUG") {
    exporter = opentelemetry::exporter::metrics::OStreamMetricExporterFactory::Create(std::cout);
  } else {
    // By default, data is export to the black hole file
    exporter = opentelemetry::exporter::metrics::OStreamMetricExporterFactory::Create(null_stream);
  }

  // Initialize and set the global MeterProvider
  opentelemetry::sdk::metrics::PeriodicExportingMetricReaderOptions reader_options;
  reader_options.export_interval_millis = std::chrono::milliseconds(export_interval_millis_);
  reader_options.export_timeout_millis = std::chrono::milliseconds(export_timeout_millis_);

  auto reader =
      opentelemetry::sdk::metrics::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), reader_options);
  auto context = opentelemetry::sdk::metrics::MeterContextFactory::Create(
      opentelemetry::sdk::metrics::ViewRegistryFactory::Create(),
      opentelemetry::sdk::resource::Resource::Create(attr_));
  context->AddMetricReader(std::move(reader));
  auto u_provider = opentelemetry::sdk::metrics::MeterProviderFactory::Create(std::move(context));
  std::shared_ptr<opentelemetry::metrics::MeterProvider> provider(std::move(u_provider));
  opentelemetry::metrics::Provider::SetMeterProvider(provider);
  // provider = opentelemetry::metrics::Provider::GetMeterProvider();
  auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("ksana_inference_metrics", SCOPE_VERSION);

  // Step 4: When creating the metrics report, you should consider using a Counter or a Histogram.
  if (metrics_.find(kForwardReqTotalNum) != metrics_.end()) {
    monitor_.call.forward_req_total_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kForwardReqTotalNum.data(), kForwardReqTotalNum.size()});
  }
  if (metrics_.find(kForwardReqErrorNum) != metrics_.end()) {
    monitor_.call.forward_req_error_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kForwardReqErrorNum.data(), kForwardReqErrorNum.size()});
  }
  if (metrics_.find(kForwardReqTimeoutNum) != metrics_.end()) {
    monitor_.call.forward_req_timeout_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kForwardReqTimeoutNum.data(), kForwardReqTimeoutNum.size()});
  }
  if (metrics_.find(kForwardReqAbortedNum) != metrics_.end()) {
    monitor_.call.forward_req_aborted_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kForwardReqAbortedNum.data(), kForwardReqAbortedNum.size()});
  }
  if (metrics_.find(kForwardCostTimeMs) != metrics_.end()) {
    monitor_.call.forward_cost_time_ms = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kForwardCostTimeMs.data(), kForwardCostTimeMs.size()});
  }
  if (metrics_.find(kPrefixCacheHitReqNum) != metrics_.end()) {
    monitor_.call.prefix_cache_hit_req_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kPrefixCacheHitReqNum.data(), kPrefixCacheHitReqNum.size()});
  }
  if (metrics_.find(kPrefixCacheHitTokenNum) != metrics_.end()) {
    monitor_.call.prefix_cache_hit_token_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kPrefixCacheHitTokenNum.data(), kPrefixCacheHitTokenNum.size()});
  }
  if (metrics_.find(kPrefixCacheHitBlockNum) != metrics_.end()) {
    monitor_.call.prefix_cache_hit_block_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kPrefixCacheHitBlockNum.data(), kPrefixCacheHitBlockNum.size()});
  }
  if (metrics_.find(kFullPromptMatchedReqNum) != metrics_.end()) {
    monitor_.call.full_prompt_matched_req_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kFullPromptMatchedReqNum.data(), kFullPromptMatchedReqNum.size()});
  }
  if (metrics_.find(kFullPromptMatchedBlockNum) != metrics_.end()) {
    monitor_.call.full_prompt_matched_block_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kFullPromptMatchedBlockNum.data(), kFullPromptMatchedBlockNum.size()});
  }
  if (metrics_.find(kBatchSchedulerBatchSize) != metrics_.end()) {
    monitor_.call.batch_scheduler_batch_size = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kBatchSchedulerBatchSize.data(), kBatchSchedulerBatchSize.size()});
  }
  if (metrics_.find(kBatchSchedulerWaitingSize) != metrics_.end()) {
    monitor_.call.batch_scheduler_waiting_size = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kBatchSchedulerWaitingSize.data(), kBatchSchedulerWaitingSize.size()});
  }
  if (metrics_.find(kBatchSchedulerSwappedSize) != metrics_.end()) {
    monitor_.call.batch_scheduler_swapped_size = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kBatchSchedulerSwappedSize.data(), kBatchSchedulerSwappedSize.size()});
  }
  if (metrics_.find(kBatchManagerScheduleMs) != metrics_.end()) {
    monitor_.call.batch_manager_schedule_ms = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kBatchManagerScheduleMs.data(), kBatchManagerScheduleMs.size()});
  }
  if (metrics_.find(kReqTotalCostInQueueMs) != metrics_.end()) {
    monitor_.call.req_total_cost_in_queue_ms = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kReqTotalCostInQueueMs.data(), kReqTotalCostInQueueMs.size()});
  }
  if (metrics_.find(kTokenNumInBatch) != metrics_.end()) {
    monitor_.call.token_num_in_batch = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kTokenNumInBatch.data(), kTokenNumInBatch.size()});
  }
  if (metrics_.find(kTokenFillRatio) != metrics_.end()) {
    monitor_.call.token_fill_ratio = meter->CreateDoubleHistogram(
        opentelemetry::v1::nostd::string_view{kTokenFillRatio.data(), kTokenFillRatio.size()});
  }
  if (metrics_.find(kBlockNumFree) != metrics_.end()) {
    monitor_.call.block_num_free =
        meter->CreateUInt64Histogram(opentelemetry::v1::nostd::string_view{kBlockNumFree.data(), kBlockNumFree.size()});
  }
  if (metrics_.find(kBlockNumUsed) != metrics_.end()) {
    monitor_.call.block_num_used =
        meter->CreateUInt64Histogram(opentelemetry::v1::nostd::string_view{kBlockNumUsed.data(), kBlockNumUsed.size()});
  }
  if (metrics_.find(kBatchSchedulerPendingSwapInSize) != metrics_.end()) {
    monitor_.call.batch_scheduler_pending_swapin_size =
        meter->CreateUInt64Counter(opentelemetry::v1::nostd::string_view{kBatchSchedulerPendingSwapInSize.data(),
                                                                         kBatchSchedulerPendingSwapInSize.size()});
  }
  if (metrics_.find(kBatchSchedulerPendingSwapOutSize) != metrics_.end()) {
    monitor_.call.batch_scheduler_pending_swapout_size =
        meter->CreateUInt64Counter(opentelemetry::v1::nostd::string_view{kBatchSchedulerPendingSwapOutSize.data(),
                                                                         kBatchSchedulerPendingSwapOutSize.size()});
  }
  if (metrics_.find(kTimeToFirstTokenMs) != metrics_.end()) {
    monitor_.call.time_to_first_token_ms = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kTimeToFirstTokenMs.data(), kTimeToFirstTokenMs.size()});
  }
  if (metrics_.find(kTimeToPerOutputTokenMs) != metrics_.end()) {
    monitor_.call.time_to_per_output_token_ms = meter->CreateDoubleHistogram(
        opentelemetry::v1::nostd::string_view{kTimeToPerOutputTokenMs.data(), kTimeToPerOutputTokenMs.size()});
  }
  if (metrics_.find(kInputTokensNum) != metrics_.end()) {
    monitor_.call.metric_input_tokens_num = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kInputTokensNum.data(), kInputTokensNum.size()});
  }
  if (metrics_.find(kOutputTokenNum) != metrics_.end()) {
    monitor_.call.metric_output_token_num = meter->CreateUInt64Histogram(
        opentelemetry::v1::nostd::string_view{kOutputTokenNum.data(), kOutputTokenNum.size()});
  }

  if (metrics_.find(kZeroOutputTokenNum) != metrics_.end()) {
    monitor_.call.metric_zero_output_token_num = meter->CreateUInt64Counter(
        opentelemetry::v1::nostd::string_view{kZeroOutputTokenNum.data(), kZeroOutputTokenNum.size()});
  }
}

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> Profiler::GetMeter(std::string meter_name) {
  auto provider = opentelemetry::metrics::Provider::GetMeterProvider();
  return provider->GetMeter(meter_name, SCOPE_VERSION);
}

void Profiler::CleanupMetrics() {
  std::shared_ptr<opentelemetry::metrics::MeterProvider> none;
  opentelemetry::metrics::Provider::SetMeterProvider(none);
}

}  // namespace ksana_llm
