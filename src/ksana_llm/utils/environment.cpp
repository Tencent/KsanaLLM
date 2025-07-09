/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/environment.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "ksana_llm/utils/memory_utils.h"

#include "ksana_llm/models/chatglm/chatglm_config.h"
#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_config.h"
#include "ksana_llm/models/gpt/gpt_config.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

DEFINE_string(config_file, "examples/ksana_llm.yaml", "The config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");

namespace ksana_llm {

void PrepareKVScales(const std::string &model_dir, ModelConfig &model_config) {
  // Search for the optional kv_cache_scales.json file
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  // TODO(zhongzhicao): 当前仅尝试从模型文件夹下读取，后续需要从python_dir/kv_scales下读取，并校验模型是否相同
  std::string &kv_scale_path = optional_file->GetOptionalFile(model_dir, "kv_scales", "kv_cache_scales.json");
  if (kv_scale_path == "") {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors file error. File not found. Using defalt value 1.0 ");
    return;
  }
  KLLM_LOG_INFO << fmt::format("Found KV cache scaling factors file at {}.", kv_scale_path);

  nlohmann::json kv_scale_json;
  std::ifstream kv_scale_file(kv_scale_path);
  if (!kv_scale_file.is_open()) {
    // TODO(zhongzhicao): load kv scale from model weights
    KLLM_LOG_WARNING << fmt::format("Failed opening KV cache scaling factors file: {}. Using defalt value 1.0 ",
                                    kv_scale_path);
  } else {
    kv_scale_file >> kv_scale_json;
    kv_scale_file.close();
  }

  uint32_t num_layers = kv_scale_json.at("kv_cache").at("scaling_factor").at("0").size();
  // TODO(zhongzhicao): 进行简单校验，后续移除
  if (model_config.num_layer != num_layers) {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors error, layer num not aligned. Using "
        "default value 1.0.");
    return;
  }

  // TODO(zhongzhicao): load kv scale for tensor_para_size > 1
  size_t tensor_parallel_size_kv_ = kv_scale_json.at("kv_cache").at("scaling_factor").size();
  if (tensor_parallel_size_kv_ != 1) {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors from TP=0. Currently only tp_size = 1 is supported.");
  }
  for (uint32_t i = 0; i < model_config.num_layer; ++i) {
    model_config.k_scales[i] = model_config.v_scales[i] =
        kv_scale_json.at("kv_cache").at("scaling_factor").at("0").at(std::to_string(i));
  }

  KLLM_LOG_INFO << fmt::format(
      "Successfully Loaded KV cache scaling factors. Currently K and V are using the same scaling factors.");
}

Environment::Environment() {}

void Environment::Reset() {
  model_config_initialized_ = false;
  model_config_ = {};
  yaml_gptq_backend_ = "";
  yaml_weight_quant_method_ = "";
  embed_tokens_use_cpu_ = false;
  is_version_report_ = true;
  cuda_graph_ = false;
  profiler_config_ = {};
  attn_backend_config_ = {};
  schedule_config_parser_.Reset();
}

Status Environment::ParseConfig(const std::string &config_file, const std::string &model_dir_override,
                                const std::string &model_config_filename) {
  Reset();
  YamlReader yaml_reader;
  const Status status = yaml_reader.LoadFile(config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Load yaml config error." << status.GetMessage();
    return status;
  }
  schedule_config_parser_.ParseConfig(yaml_reader);

  // Read profiler config.
  profiler_config_.trace_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.trace_export_url", "");
  profiler_config_.metrics_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.metrics_export_url", "");
  profiler_config_.export_interval_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_interval_millis", 60000);
  profiler_config_.export_timeout_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_timeout_millis", 1000);

  auto attributes = yaml_reader.GetMap(yaml_reader.GetRootNode(), "setting.profiler.attributes");
  for (auto it = attributes.begin(); it != attributes.end(); ++it) {
    const std::string &key = it->first.as<std::string>();
    const std::string &value = it->second.as<std::string>();
    profiler_config_.resource_attributes[key] = value;
  }
  // quantization_config in yaml takes effect when quantization_config in
  // config.json is null.
  yaml_weight_quant_method_ = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.quantization_config.weight.quant_method", "auto");

  yaml_gptq_backend_ = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                          "setting.quantization_config.gptq_backend", "cutlass");

  // Read attn backend config.
  attn_backend_config_.enable_blocked_multi_token_forwarding_kv = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.attn_backend.enable_blocked_multi_token_forwarding_kv", false);
  KLLM_LOG_INFO << "enable_blocked_multi_token_forwarding_kv: "
                << attn_backend_config_.enable_blocked_multi_token_forwarding_kv;

  // Read base model.
  std::string base_model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  std::string tokenizer_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.tokenizer_dir", "");
  if (tokenizer_dir.empty()) {
    tokenizer_dir = base_model_dir;
  }
  if (!model_dir_override.empty()) {
    base_model_dir = model_dir_override;
    tokenizer_dir = model_dir_override;
  }
  STATUS_CHECK_RETURN(ParseModelConfig(yaml_reader, base_model_dir, tokenizer_dir, model_config_filename));

  return Status();
}

void Environment::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  schedule_config_parser_.SetReservedDeviceRatio(reserved_device_memory_ratio);
}

Status Environment::UpdateModelConfig() {
  KLLM_CHECK_WITH_INFO(model_config_initialized_, "model_config not initialized.");
  return schedule_config_parser_.UpdateModelConfig(model_config_);
}

Status Environment::ParseModelConfig(YamlReader &yaml_reader, const std::string &model_dir,
                                     const std::string &tokenizer_dir, const std::string &model_config_filename) {
  KLLM_CHECK_WITH_INFO(!model_config_initialized_, "model_config_initialized_ initialized.");
  EnvModelConfigParser model_config_parser(yaml_weight_quant_method_, yaml_gptq_backend_);
  model_config_parser.ParseModelConfig(model_dir, tokenizer_dir, model_config_filename, model_config_);
  schedule_config_parser_.UpdateModelConfig(model_config_);

  auto kv_cache_dtype_str = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                               "setting.quantization_config.kv_cache.dtype", "auto");
  DataType kv_cache_dtype = model_config_.weight_data_type;

  if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv &&
      !(schedule_config_parser_.IsFlashMlaEnable() && !IsPrefixCachingEnabled())) {
    if (kv_cache_dtype_str == "fp8_e5m2" || kv_cache_dtype_str == "fp8_e4m3") {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
  } else {
    if (kv_cache_dtype_str == "fp8_e5m2") {
      kv_cache_dtype = TYPE_FP8_E5M2;
    } else if (kv_cache_dtype_str == "fp8_e4m3") {
      kv_cache_dtype = TYPE_FP8_E4M3;
      PrepareKVScales(model_dir, model_config_);
    }
  }
  schedule_config_parser_.UpdateMembers(model_config_, kv_cache_dtype);

  model_config_initialized_ = true;

  KLLM_LOG_INFO << fmt::format(
      "Load model {} from config file: {} success. num_layer={}, "
      "hidden_units={}, head_num={}, vocab_size={}",
      model_config_.name, model_config_.path + "/" + model_config_filename, model_config_.num_layer,
      model_config_.hidden_units, model_config_.head_num, model_config_.vocab_size);
  return Status();
}

Status Environment::GetModelConfig(ModelConfig &model_config) {
  if (!model_config_initialized_) {
    return Status(RET_MODEL_INVALID, "model config not initialized");
  }
  model_config = model_config_;
  return Status();
}

Status Environment::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  return schedule_config_parser_.GetBatchSchedulerConfig(batch_scheduler_config);
}

Status Environment::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  return schedule_config_parser_.GetCacheManagerConfig(cache_manager_config);
}

Status Environment::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  return schedule_config_parser_.GetBlockManagerConfig(block_manager_config);
}

void Environment::SetBlockManagerConfig(const BlockManagerConfig &block_manager_config) {
  schedule_config_parser_.SetBlockManagerConfig(block_manager_config);
}

std::tuple<size_t, size_t> Environment::GetCacheBlockSize(const ModelConfig &model_config,
                                                          const PipelineConfig &pipeline_config,
                                                          const BlockManagerConfig &block_manager_config) {
  return schedule_config_parser_.GetCacheBlockSize(model_config, pipeline_config, block_manager_config);
}

Status Environment::CalculateBlockNumber() { return schedule_config_parser_.CalculateBlockNumber(); }

Status Environment::ResetPipelineBlockNumber() { return schedule_config_parser_.ResetPipelineBlockNumber(); }

size_t Environment::GetBlockTokenNum() { return schedule_config_parser_.GetBlockTokenNum(); }

size_t Environment::GetConvertSize() { return schedule_config_parser_.GetConvertSize(); }

size_t Environment::GetBlockSize() { return schedule_config_parser_.GetBlockSize(); }

size_t Environment::GetTotalDeviceBlockNum() { return schedule_config_parser_.GetTotalDeviceBlockNum(); }

size_t Environment::GetTotalHostBlockNum() { return schedule_config_parser_.GetTotalHostBlockNum(); }

DataType Environment::GetKVCacheType() { return schedule_config_parser_.GetKVCacheType(); }

std::vector<int> Environment::GetDataParaGroupDevices(int dp_id) {
  return schedule_config_parser_.GetDataParaGroupDevices(dp_id);
}

size_t Environment::GetAttentionTensorParallel() { return schedule_config_parser_.GetAttentionTensorParallel(); }

void Environment::SetDataParaGroupStatus(int dp_group_id, bool enabled) {
  schedule_config_parser_.SetDataParaGroupStatus(dp_group_id, enabled);
}

bool Environment::GetDataParaGroupStatus(int dp_group_id) {
  return schedule_config_parser_.GetDataParaGroupStatus(dp_group_id);
}

Status Environment::GetProfilerConfig(ProfilerConfig &profiler_config) {
  profiler_config = profiler_config_;
  return Status();
}

bool Environment::IsPrefixCachingEnabled() { return schedule_config_parser_.IsPrefixCachingEnabled(); }

bool Environment::IsFlexibleCachingEnabled() { return schedule_config_parser_.IsFlexibleCachingEnabled(); }

bool Environment::IsSpeculativeDecodingEnabled() { return schedule_config_parser_.IsSpeculativeDecodingEnabled(); }

bool Environment::IsPrefillDecodeSeparation() { return schedule_config_parser_.IsPrefillDecodeSeparation(); }

bool Environment::IsMTPEnabled() { return schedule_config_parser_.IsMTPEnabled(); }

size_t Environment::GetTransferLayerChunkSize() { return schedule_config_parser_.GetTransferLayerChunkSize(); }

}  // namespace ksana_llm
