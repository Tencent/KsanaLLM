/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once
#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif
#include <unistd.h>

#include <any>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/connector/config.h"
#include "ksana_llm/utils/config/model_config_parser.h"
#include "ksana_llm/utils/config/schedule_config_parser.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace ksana_llm {
constexpr size_t kStepGenerateTokenNum = 1;  // The number of tokens that the model generated at each step

// The endpoint type.
enum class EndpointType { LOCAL, RPC };

// The config of endpoint.
struct EndpointConfig {
  // The endpoint service type.
  EndpointType type = EndpointType::LOCAL;

  // If the endpoint type is RPC, load the corresponding
  // shared library based on the rpc plugin name.
  std::string rpc_plugin_name;

  // The endpoint service host address.
  std::string host = "0.0.0.0";

  // The endpoint service port.
  uint32_t port = 8080;

  // Whether to enable the endpoint access log.
  bool access_log = true;
};

// The config of profiler.
struct ProfilerConfig {
  // The stat interval, in second.
  std::string trace_export_url;
  std::string metrics_export_url;
  uint64_t export_interval_millis;
  uint64_t export_timeout_millis;

  // Opentelemetry Resource attributes.
  std::unordered_map<std::string, std::string> resource_attributes;
};

class Environment {
 public:
  Environment();

  // Parse environment from YAML config file.
  // model_dir_override is used to override model_dir and tokenizer_dir in config_file
  Status ParseConfig(const std::string &config_file, const std::string &model_dir_override = "",
                     const std::string &model_config_filename = "config.json");

  // Get the model config
  Status GetModelConfig(ModelConfig &model_config);

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  Status GetRuntimeConfig(RuntimeConfig &runtime_config);

  size_t GetTransferLayerChunkSize();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // TODO(yancyliu): remove from here later.
  Status CalculateBlockNumber();
  Status ResetPipelineBlockNumber();
  size_t GetTotalDeviceBlockNum();
  size_t GetTotalHostBlockNum();
  std::vector<int> GetDataParaGroupDevices(int dp_id);

  // Get the config of profiler.
  Status GetProfilerConfig(ProfilerConfig &profiler_config);

  const std::string &GetYamlGptqBackend() const { return yaml_gptq_backend_; }

  const std::string &GetYamlWeightQuantMethod() const { return yaml_weight_quant_method_; }

  bool IsReportVersion() { return is_version_report_; }

  size_t GetMaxBatchSize() const { return schedule_config_parser_.GetMaxBatchSize(); }

  Status GetPipelineConfig(PipelineConfig &pipeline_config) const {
    return schedule_config_parser_.GetPipelineConfig(pipeline_config);
  }

  Status GetExpertParallelConfig(ExpertParallelConfig &expert_parallel_config) const {
    return schedule_config_parser_.GetExpertParallelConfig(expert_parallel_config);
  }

  Status GetConnectorConfigs(ConnectorConfig &connector_config) const {
    return schedule_config_parser_.GetConnectorConfigs(connector_config);
  }

  // Calculate block size via model configs.
  Status InitializeBlockManagerConfig() { return schedule_config_parser_.InitializeBlockManagerConfig(model_config_); }

  // Init Expert-Parallel Config from env.
  void InitializeExpertParallelConfig() { schedule_config_parser_.InitializeExpertParallelConfig(); }

 public:  // Used for test.
  // Update model config after batch scheduler config is updated.
  // TODO(robertyuan): remove later.
  Status UpdateModelConfig();

  void SetModelConfig(ModelConfig &model_config) { model_config_ = model_config; }
  void SetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config) {
    schedule_config_parser_.SetBatchSchedulerConfig(batch_manager_config);
  }
  void SetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
    schedule_config_parser_.SetCacheManagerConfig(cache_manager_config);
  }

  void SetTensorParallelSize(size_t tensor_parallel_size) {
    schedule_config_parser_.SetTensorParallelSize(tensor_parallel_size);
  }

  void SetExpertParallelConfig(const ExpertParallelConfig &expert_parallel_config) {
    return schedule_config_parser_.SetExpertParallelConfig(expert_parallel_config);
  }

  void GetAttnBackendConfig(AttnBackendConfig &attn_backend_config) {
    schedule_config_parser_.GetAttnBackendConfig(attn_backend_config);
  }
  void SetAttnBackendConfig(const AttnBackendConfig &attn_backend_config) {
    schedule_config_parser_.SetAttnBackendConfig(attn_backend_config);
  }

  void SetConnectorConfigs(const ConnectorConfig &connector_config) {
    schedule_config_parser_.SetConnectorConfigs(connector_config);
  }

  // Modify reserved_device_memory_ratio
  void SetReservedDeviceRatio(float reserved_device_memory_ratio);

  // Set and get multiple node pipeline config.
  void SetPipelineConfig(const PipelineConfig &pipeline_config) {
    return schedule_config_parser_.SetPipelineConfig(pipeline_config);
  }

  void SetAttnDataParallelSize(size_t attn_data_parallel_size) {
    schedule_config_parser_.SetAttnDataParallelSize(attn_data_parallel_size);
  }

  void SetExpertParallelSize(size_t expert_parallel_size) {
    schedule_config_parser_.SetExpertParallelSize(expert_parallel_size);
  }

  void SetBlockManagerConfig(const BlockManagerConfig &block_manager_config);

  size_t GetCacheBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                           const BlockManagerConfig &block_manager_config);

 private:
  void Reset();

  // Parse model config from model dir.
  Status ParseModelAndScheduleConfig(YamlReader &yaml_reader, const std::string &model_dir,
                                     const std::string &tokenizer_dir, const std::string &model_config_filename);

 private:
  bool model_config_initialized_;
  ModelConfig model_config_;

  // The backend of gptq/awq quantization.
  std::string yaml_gptq_backend_;

  // The config of quantization.
  std::string yaml_weight_quant_method_;

  // The config of profiler.
  ProfilerConfig profiler_config_;

  bool is_version_report_ = true;

  ScheduleConfigParser schedule_config_parser_;
};

}  // namespace ksana_llm
