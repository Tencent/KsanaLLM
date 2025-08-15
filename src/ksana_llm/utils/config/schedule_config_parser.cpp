/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/config/schedule_config_parser.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"

#include "ksana_llm/models/chatglm/chatglm_config.h"
#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_config.h"
#include "ksana_llm/models/gpt/gpt_config.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/dynamic_memory_counter.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

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

ScheduleConfigParser::ScheduleConfigParser() { Reset(); }

size_t ScheduleConfigParser::GetCommonBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                                const BlockManagerConfig &block_manager_config) {
  const bool predict_nextn =
      static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
  const size_t node_nextn_layer_num =
      predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;
  const size_t node_layer_num =
      pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

  const size_t token_size =
      node_layer_num *
      (model_config.num_key_value_heads / (runtime_config_.parallel_basic_config.tensor_parallel_size /
                                           runtime_config_.parallel_basic_config.attn_data_parallel_size)) *
      model_config.size_per_head;
  const size_t block_token_num = block_manager_config.device_allocator_config.block_token_num;
  const size_t block_dtype_size = GetTypeSize(block_manager_config.device_allocator_config.kv_cache_dtype);

  const size_t cache_block_size = token_size * block_token_num * 2 * block_dtype_size;
  KLLM_LOG_INFO << fmt::format("Init block num for key or value: ({} / {}) * ({} / {}) * {} = {}", node_layer_num, 1,
                               model_config.num_key_value_heads,
                               (runtime_config_.parallel_basic_config.tensor_parallel_size /
                                runtime_config_.parallel_basic_config.attn_data_parallel_size),
                               model_config.size_per_head, token_size);

  KLLM_LOG_INFO << fmt::format("Init token size (bytes) of init block for both key and value: {} * {} * 2 * {} = {}",
                               token_size, block_manager_config.device_allocator_config.block_token_num,
                               GetTypeSize(block_manager_config.device_allocator_config.kv_cache_dtype),
                               cache_block_size);
  return cache_block_size;
}

size_t ScheduleConfigParser::GetDeepSeekV3BlockSize(const ModelConfig &model_config,
                                                    const PipelineConfig &pipeline_config,
                                                    const BlockManagerConfig &block_manager_config) {
  const bool predict_nextn =
      static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
  const size_t node_nextn_layer_num =
      predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;
  const size_t node_layer_num =
      pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

  // For MLA, compressed of kv-cache is (kv_lora_rank + qk_rope_head_dim), and it have only one head.
  const size_t token_size =
      node_layer_num * (model_config.mla_config.kv_lora_rank + model_config.mla_config.qk_rope_head_dim);
  const size_t block_token_num = block_manager_config.device_allocator_config.block_token_num;
  const size_t block_dtype_size = GetTypeSize(block_manager_config.device_allocator_config.kv_cache_dtype);

  const size_t cache_block_size = token_size * block_token_num * block_dtype_size;
  KLLM_LOG_INFO << fmt::format(
      "Init cache block size, node_layer_num:{}, kv_lora_rank:{}, qk_rope_head_dim:{}, block_token_num:{}, "
      "cache_block_size:{}.",
      node_layer_num, model_config.mla_config.kv_lora_rank, model_config.mla_config.qk_rope_head_dim, block_token_num,
      cache_block_size);

  return cache_block_size;
}

size_t ScheduleConfigParser::GetCacheBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                               const BlockManagerConfig &block_manager_config) {
  if (model_config.type == "deepseek_v2" || model_config.type == "deepseek_v3") {
    if (IsAbsorbWeightsEnabled()) {
      return GetDeepSeekV3BlockSize(model_config, pipeline_config, block_manager_config);
    }
    return GetCommonBlockSize(model_config, pipeline_config, block_manager_config);
  }

  return GetCommonBlockSize(model_config, pipeline_config, block_manager_config);
}

void ScheduleConfigParser::Reset() {
  batch_scheduler_config_ = {};
  cache_manager_config_ = {};
  block_manager_config_ = {};
  pipeline_config_ = {};
  expert_parallel_config_ = {};
  connector_config_ = {};
  runtime_config_ = {};
}

Status ScheduleConfigParser::ParseScheduleConfig(YamlReader &yaml_reader, ModelConfig &model_config) {
  // Read global setting.
  runtime_config_.parallel_basic_config.tensor_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.tensor_para_size", 0);
  runtime_config_.parallel_basic_config.attn_data_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.attn_data_para_size", 1);
  runtime_config_.parallel_basic_config.expert_world_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  runtime_config_.parallel_basic_config.expert_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);
  runtime_config_.enable_full_shared_expert =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_full_shared_expert", false);
  if (runtime_config_.parallel_basic_config.tensor_parallel_size == 0) {
    int device_size = -1;
    GetDeviceCount(&device_size);
    runtime_config_.parallel_basic_config.tensor_parallel_size = static_cast<size_t>(device_size);
  }

  KLLM_CHECK_WITH_INFO(
      runtime_config_.parallel_basic_config.tensor_parallel_size >=
          runtime_config_.parallel_basic_config.attn_data_parallel_size,
      fmt::format("Tensor Para Size(tensor_para_size) {} should >= Attention Data Para Size(attn_data_para_size) {}",
                  runtime_config_.parallel_basic_config.tensor_parallel_size,
                  runtime_config_.parallel_basic_config.attn_data_parallel_size));

  KLLM_CHECK_WITH_INFO(
      runtime_config_.parallel_basic_config.tensor_parallel_size %
              runtime_config_.parallel_basic_config.attn_data_parallel_size ==
          0,
      fmt::format("Tensor Para Size(tensor_para_size) {} % Attention Data Para Size(attn_data_para_size) {} != 0",
                  runtime_config_.parallel_basic_config.tensor_parallel_size,
                  runtime_config_.parallel_basic_config.attn_data_parallel_size));

#if (defined(ENABLE_ACL) || defined(ENABLE_TOPS))
  if (runtime_config_.parallel_basic_config.attn_data_parallel_size > 1) {
    KLLM_THROW(
        fmt::format("Huawei Ascend does not support data parallelism, please set attn_data_parallel_size to 1."));
  }
#endif
  if (!(runtime_config_.parallel_basic_config.tensor_parallel_size > 0 &&
        runtime_config_.parallel_basic_config.attn_data_parallel_size > 0)) {
    KLLM_THROW(fmt::format("Tensor Para Size {}, Data Para Size {} should > 0",
                           runtime_config_.parallel_basic_config.tensor_parallel_size,
                           runtime_config_.parallel_basic_config.attn_data_parallel_size));
  }

  int device_num;
  GetDeviceCount(&device_num);
  KLLM_CHECK_WITH_INFO(device_num >= runtime_config_.parallel_basic_config.tensor_parallel_size,
                       fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                   runtime_config_.parallel_basic_config.tensor_parallel_size, device_num));

  // Get each atten data parallel group size.
  // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the number of
  // attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp = 2, then each
  // attn dp group size is 2.
  runtime_config_.parallel_basic_config.attn_tensor_parallel_size =
      runtime_config_.parallel_basic_config.tensor_parallel_size /
      runtime_config_.parallel_basic_config.attn_data_parallel_size;

  // NOTE(karlluo): When using PP parallelism (pipeline parallelism), the communication mode is selected, with the
  // default value being "default". The "default" mode is the send-receive mode. When node0 completes the inference of
  // the previous task, device0 on node0 sends data to device0 on node1, and device1 on node0 sends data to device1 on
  // node1. The "scatter" mode is the scatter mode. When node0 completes the inference of the previous task, device0 on
  // node0 sends data to device0, device1, device2, etc., on node1.
  const std::string &pp_comm_type_str = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.global.pipeline_para_comm_type", "default");
  if (pp_comm_type_str == "scatter") {
    pipeline_config_.pipeline_para_comm_type = DistributedCommunicationType::SCATTER;
  }

  // Read batch scheduler config.
  batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.schedule_strategy", 0));
  batch_scheduler_config_.pp_multibatch_wb_strategy = static_cast<PPMultibatchWBStrategy>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.pp_multibatch_wb_strategy", 0));
  batch_scheduler_config_.waiting_timeout_in_ms =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.waiting_timeout_in_ms", 600000);
  batch_scheduler_config_.max_waiting_queue_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_waiting_queue_len", 1200);
  batch_scheduler_config_.max_token_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_token_len", 0);
  batch_scheduler_config_.max_step_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_step_tokens", 4096);
  batch_scheduler_config_.max_batch_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_batch_size", 128);
  batch_scheduler_config_.max_pretransfer_batch_size = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.max_pretransfer_batch_size", 64);
  batch_scheduler_config_.transfer_layer_chunk_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.transfer_layer_chunk_size", 1);
  batch_scheduler_config_.max_pp_batch_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_pp_batch_num", 1);
  batch_scheduler_config_.swapout_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapout_block_threshold", 1.0);
  batch_scheduler_config_.swapin_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapin_block_threshold", 2.0);
  batch_scheduler_config_.launch_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.launch_block_threshold", 2.0);
  batch_scheduler_config_.preempt_mode = static_cast<PreemptMode>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.preempt_mode", 0));
  batch_scheduler_config_.split_fuse_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.split_fuse_token_num", 0);
  batch_scheduler_config_.enable_speculative_decoding = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_speculative_decoding", false);
  batch_scheduler_config_.enable_mtp_module =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_mtp_module", false);
  batch_scheduler_config_.enable_xgrammar =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_xgrammar", false);

  KLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_pp_batch_num > 0, "max_multi_batch_size should be bigger than 0");

  // When MTP is enabled, each request requires calculating 2 tokens while decoding.
  batch_scheduler_config_.max_decode_tokens_per_req = batch_scheduler_config_.enable_mtp_module ? 2 : 1;

  if (runtime_config_.parallel_basic_config.attn_data_parallel_size > 1) {
    KLLM_CHECK_WITH_INFO(
        batch_scheduler_config_.max_step_token_num / runtime_config_.parallel_basic_config.attn_data_parallel_size >=
            batch_scheduler_config_.max_token_len,
        fmt::format("max_step_token_num({}) / attn_data_para_size({}) must >= max_token_len({})",
                    batch_scheduler_config_.max_step_token_num,
                    runtime_config_.parallel_basic_config.attn_data_parallel_size,
                    batch_scheduler_config_.max_token_len));
  }

  // Read block manager config.
  block_manager_config_.host_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.device_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);

  // If DeepSeek model, automatically enable Flash MLA and set block_token_num to 64.
  if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
    block_manager_config_.host_allocator_config.block_token_num = 64;
    block_manager_config_.device_allocator_config.block_token_num = 64;
    KLLM_LOG_INFO
        << "Automatically activate Flash MLA for DeepSeek models, setting block_token_num to 64 for flash_mla";
  }
  block_manager_config_.reserved_device_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.reserved_device_memory_ratio", 0.01);
  block_manager_config_.block_device_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_device_memory_ratio", -1.0);
  block_manager_config_.block_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_host_memory_factor", 2.0);
  block_manager_config_.dynamic_reusable_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.dynamic_reusable_memory_ratio", 1.0);

  // Load cache manager config
  cache_manager_config_.swap_threadpool_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swap_threadpool_size", 2);
  cache_manager_config_.min_flexible_cache_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.min_flexible_cache_num", 0);
  cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  cache_manager_config_.tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
  cache_manager_config_.enable_prefix_caching =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_auto_prefix_cache", false);
#ifdef ENABLE_ACL
  if (cache_manager_config_.enable_prefix_caching) {
    cache_manager_config_.enable_prefix_caching = false;
    KLLM_LOG_WARNING << "prefix caching not support NPU, will change enable_prefix_caching as false";
  }
#endif
  // TODO(zakwang): Implement support for cases where prefix caching is disabled while split_fuse_token_num is non-zero.
  if (batch_scheduler_config_.split_fuse_token_num != 0 && !cache_manager_config_.enable_prefix_caching) {
    KLLM_LOG_WARNING << "While prefix caching is disabled，split_fuse_token_num will always be disabled. So set "
                        "split_fuse_token_num to 0.";
    batch_scheduler_config_.split_fuse_token_num = 0;
  }

  // Read parallel config.
  expert_parallel_config_.expert_world_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  expert_parallel_config_.expert_para_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);

  // Read attn backend config.
  runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.attn_backend.enable_blocked_multi_token_forwarding_kv", false);
  KLLM_LOG_INFO << "enable_blocked_multi_token_forwarding_kv: "
                << runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv;

  InitConnectorConfig(yaml_reader);
  return Status();
}

void ScheduleConfigParser::UpdateMembers(const std::string &model_dir, ModelConfig &model_config,
                                         std::string &kv_cache_dtype_str) {
  DataType kv_cache_dtype = model_config.weight_data_type;

  if (runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv && IsPrefixCachingEnabled()) {
    if (kv_cache_dtype_str == "fp8_e5m2" || kv_cache_dtype_str == "fp8_e4m3") {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
  } else {
    if (kv_cache_dtype_str == "fp8_e5m2") {
#ifdef ENABLE_CUDA
      if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
        KLLM_LOG_WARNING << "Flash MLA not support fp8_e5m2 KV Cache. Please use fp8_e4m3.";
      }
#endif
      kv_cache_dtype = TYPE_FP8_E5M2;
    } else if (kv_cache_dtype_str == "fp8_e4m3") {
      kv_cache_dtype = TYPE_FP8_E4M3;
      PrepareKVScales(model_dir, model_config);
    }
  }

  if (model_config.is_quant == true && model_config.quant_config.method == QUANT_FP8_E4M3 &&
      model_config.quant_config.is_checkpoint_fp8_serialized == false) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is fp8_e4m3, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  } else if (model_config.is_quant == true && model_config.quant_config.method == QUANT_GPTQ) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is gptq, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  }

  block_manager_config_.host_allocator_config.kv_cache_dtype = kv_cache_dtype;
  block_manager_config_.device_allocator_config.kv_cache_dtype = kv_cache_dtype;
  runtime_config_.attn_backend_config.kv_cache_dtype = block_manager_config_.device_allocator_config.kv_cache_dtype;
}

void ScheduleConfigParser::InitConnectorConfig(
    YamlReader &yaml_reader) {  // Parse connector role first to check if we should continue parsing
  std::string role_str =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.group_role", "none");

  const bool decode_node_benchmark =
      (std::getenv("DECODE_NODE_BENCHMARK") != nullptr) && (strcmp(std::getenv("DECODE_NODE_BENCHMARK"), "1") == 0);
  if (decode_node_benchmark) {
    role_str = "decode";
  }

  // Convert to lowercase for case-insensitive comparison
  std::transform(role_str.begin(), role_str.end(), role_str.begin(), [](unsigned char c) { return std::tolower(c); });
  // Check if the role is not None
  if (role_str != "none") {
    // Set role based on parsed string
    if (role_str == "prefill") {
      connector_config_.group_role = GroupRole::PREFILL;
    } else if (role_str == "decode") {
      connector_config_.group_role = GroupRole::DECODE;
    } else if (role_str == "both") {
      connector_config_.group_role = GroupRole::BOTH;
    } else {
      connector_config_.group_role = GroupRole::NONE;
      KLLM_LOG_WARNING << fmt::format("Unknown connector role: {}, defaulting to NONE", role_str);
    }

    // Only continue parsing if the role is not NONE
    if (connector_config_.group_role != GroupRole::NONE) {
      // Parse connector type
      connector_config_.router_endpoint =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.router_endpoint", "");
      connector_config_.group_name =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.group_name", "");
      connector_config_.node_name =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.node_name", "");
      connector_config_.heartbeat_interval_ms =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.heartbeat_interval_ms", 5000);
      connector_config_.coordinator_port =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.coordinator_port", 1357);
      connector_config_.transfer_batch =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.transfer_batch", 1048576);
      connector_config_.connector_waiting_sec =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.connector_waiting_sec", 1800);
      connector_config_.circular_bucket_size =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_bucket_size", 8192);
      connector_config_.circular_bucket_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_bucket_num", 4);
      connector_config_.circular_thread_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_thread_num", 4);
      connector_config_.send_thread_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.send_thread_num", 4);
      connector_config_.inference_addr =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.inference_addr", "");
      connector_config_.inference_port =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.inference_port", 8080);
      std::string type_str =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.communication_type", "");

      // Convert to lowercase for case-insensitive comparison
      std::transform(type_str.begin(), type_str.end(), type_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (type_str == "nccl") {
        connector_config_.communication_type = CommunicationType::NCCL;
      } else if (type_str == "zmq") {
        connector_config_.communication_type = CommunicationType::ZMQ;
      } else {
        connector_config_.communication_type = CommunicationType::TCP;
      }

      // Log the parsed configuration
      KLLM_LOG_INFO << fmt::format(
          "Connector config parsed: role={}, type={}, router_endpoint={}, group_name={}, "
          "node_name={}, heartbeat_interval={}ms",
          role_str, type_str, connector_config_.router_endpoint, connector_config_.group_name,
          connector_config_.node_name, connector_config_.heartbeat_interval_ms);
    }
  } else {
    KLLM_LOG_INFO << "Connector role is set to NONE, skipping connector configuration.";
  }

  if (decode_node_benchmark) {
    connector_config_.router_endpoint = "decode_benchmark";
  }
}

void ScheduleConfigParser::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  block_manager_config_.reserved_device_memory_ratio = reserved_device_memory_ratio;
}

Status ScheduleConfigParser::UpdateModelConfig(ModelConfig &model_config) {
  if (cache_manager_config_.min_flexible_cache_num != 0 && model_config.use_qk_norm) {
    cache_manager_config_.min_flexible_cache_num = 0;
    KLLM_LOG_WARNING << "flexible cache and qk norm cannot be used together, set min_flexible_cache_num to 0";
  }

  if (runtime_config_.parallel_basic_config.tensor_parallel_size > model_config.num_key_value_heads ||
      model_config.num_key_value_heads % runtime_config_.parallel_basic_config.tensor_parallel_size != 0) {
    KLLM_THROW(
        fmt::format("The size of key_value_heads cannot be evenly divided by the size of "
                    "runtime_config_.parallel_basic_config.tensor_parallel_size. "
                    "{} % {} != 0 ",
                    model_config.num_key_value_heads, runtime_config_.parallel_basic_config.tensor_parallel_size));
  }

  if (runtime_config_.parallel_basic_config.tensor_parallel_size <
          runtime_config_.parallel_basic_config.expert_parallel_size ||
      runtime_config_.parallel_basic_config.tensor_parallel_size %
              runtime_config_.parallel_basic_config.expert_parallel_size !=
          0) {
    KLLM_THROW(
        fmt::format("The size of runtime_config_.parallel_basic_config.tensor_parallel_size cannot be evenly divided "
                    "by the size of "
                    "runtime_config_.parallel_basic_config.expert_parallel_size. "
                    "{} % {} != 0 ",
                    runtime_config_.parallel_basic_config.tensor_parallel_size,
                    runtime_config_.parallel_basic_config.expert_parallel_size));
  }

  if (batch_scheduler_config_.max_token_len > 0) {
    if (batch_scheduler_config_.max_token_len > model_config.max_training_seq_len) {
      KLLM_LOG_WARNING << fmt::format(
          "The max_training_seq_len configured in the model's config.json is less than the "
          "max_token_len configured in the ksana yaml file. {} < {}, use {}",
          model_config.max_training_seq_len, batch_scheduler_config_.max_token_len, model_config.max_training_seq_len);
      runtime_config_.max_seq_len = model_config.max_training_seq_len;
    } else {
      runtime_config_.max_seq_len = batch_scheduler_config_.max_token_len;
    }
  } else {
    runtime_config_.max_seq_len = model_config.max_training_seq_len;
  }
  batch_scheduler_config_.max_token_len = runtime_config_.max_seq_len;
  if ((batch_scheduler_config_.split_fuse_token_num == 0) &&
      (batch_scheduler_config_.max_step_token_num < batch_scheduler_config_.max_token_len)) {
    // if no split fuse, request cannot be processed if max_step_num < input token num
    batch_scheduler_config_.max_step_token_num = batch_scheduler_config_.max_token_len;
  }

  runtime_config_.parallel_basic_config.moe_tensor_para_size =
      runtime_config_.parallel_basic_config.tensor_parallel_size /
      runtime_config_.parallel_basic_config.expert_parallel_size;

  runtime_config_.inter_data_type = model_config.weight_data_type;
  // TODO(robertyuan): These members should be removed from other configs
  runtime_config_.max_batch_size = batch_scheduler_config_.max_batch_size;
  runtime_config_.max_pp_batch_num = batch_scheduler_config_.max_pp_batch_num;
  runtime_config_.max_step_token_num = batch_scheduler_config_.max_step_token_num;
  runtime_config_.enable_mtp_module = batch_scheduler_config_.enable_mtp_module;
  runtime_config_.enable_speculative_decoding = batch_scheduler_config_.enable_speculative_decoding;

  runtime_config_.separate_prefill_decode = (connector_config_.group_role != GroupRole::NONE);
  runtime_config_.enable_prefix_caching = cache_manager_config_.enable_prefix_caching;
  runtime_config_.enable_flexible_caching = cache_manager_config_.min_flexible_cache_num > 0;

  runtime_config_.attn_backend_config.kv_cache_dtype = block_manager_config_.device_allocator_config.kv_cache_dtype;
  runtime_config_.attn_backend_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;

  return Status();
}

Status ScheduleConfigParser::InitializeBlockManagerConfig(const ModelConfig &model_config) {
  if (pipeline_config_.lower_layer_idx < 0 || pipeline_config_.upper_layer_idx < 0) {
    pipeline_config_.lower_layer_idx = 0;
    pipeline_config_.upper_layer_idx = model_config.num_layer - 1;
    if (model_config.num_nextn_predict_layers != 0 && batch_scheduler_config_.enable_mtp_module) {
      pipeline_config_.lower_nextn_layer_idx = model_config.num_layer;
      pipeline_config_.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
    }
  }

  // Determine block size.
  size_t cache_block_size = GetCacheBlockSize(model_config, pipeline_config_, block_manager_config_);

  block_manager_config_.host_allocator_config.block_size = cache_block_size;
  block_manager_config_.device_allocator_config.block_size = cache_block_size;

  block_manager_config_.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;

  // The default block number, will be overwrited through memory usage.
  block_manager_config_.host_allocator_config.blocks_num = 512 * 10;
  block_manager_config_.device_allocator_config.blocks_num = 512;

  runtime_config_.attn_backend_config.block_size = block_manager_config_.device_allocator_config.block_size;
  return CheckEnvironment();
}

Status ScheduleConfigParser::CheckEnvironment() {
  if (block_manager_config_.host_allocator_config.block_size !=
      block_manager_config_.device_allocator_config.block_size) {
    return Status(RET_INIT_FAILED, fmt::format("block size of device and host is not equal, {} vs {}.",
                                               block_manager_config_.host_allocator_config.block_size,
                                               block_manager_config_.device_allocator_config.block_size));
  }

  return Status();
}

Status ScheduleConfigParser::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config = batch_scheduler_config_;
  return Status();
}

void ScheduleConfigParser::SetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config_ = batch_scheduler_config;
}

Status ScheduleConfigParser::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config = cache_manager_config_;
  return Status();
}

void ScheduleConfigParser::SetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config_ = cache_manager_config;
}

Status ScheduleConfigParser::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  block_manager_config = block_manager_config_;
  return Status();
}

void ScheduleConfigParser::SetBlockManagerConfig(const BlockManagerConfig &block_manager_config) {
  block_manager_config_ = block_manager_config;
}

Status ScheduleConfigParser::GetRuntimeConfig(RuntimeConfig &runtime_config) {
  runtime_config = runtime_config_;
  return Status();
}

Status ScheduleConfigParser::CalculateBlockNumber() {
  size_t host_total, host_free;
  size_t device_total, device_free;
  float device_reserved_ratio = block_manager_config_.reserved_device_memory_ratio;

  if (!DeviceMemoryPool::Empty()) {
    device_total = DeviceMemoryPool::GetMemoryPool(0)->GetTotalByte();
    device_free = DeviceMemoryPool::GetMemoryPool(0)->GetMaxContinuousFreeByte(true);

    // Because block allocate need (block_num_ + 1* * block_size bytes, why?
    if (device_free <= block_manager_config_.device_allocator_config.block_size) {
      throw std::runtime_error(fmt::format("The device_free {} should large than block_size {}", device_free,
                                           block_manager_config_.device_allocator_config.block_size));
    }
    device_free -= block_manager_config_.device_allocator_config.block_size;
  } else {
    // Allocate blocks according to the memory status of device 0.
    SetDevice(0);
    Status status =
        GetDeviceMemoryInfo(block_manager_config_.device_allocator_config.device, &device_free, &device_total);
    if (!status.OK()) {
      return status;
    }
  }

  Status status = GetHostMemoryInfo(&host_free, &host_total);
  if (!status.OK()) {
    return status;
  }

  KLLM_LOG_INFO << "Get memory info, host_total:" << host_total << ", host_free:" << host_free
                << ", device_total:" << device_total << ", device_free:" << device_free
                << ", block_device_memory_ratio:" << block_manager_config_.block_device_memory_ratio
                << ", reserved_device_memory_ratio:" << block_manager_config_.reserved_device_memory_ratio
                << ", block_host_memory_factor:" << block_manager_config_.block_host_memory_factor;

  KLLM_CHECK_WITH_INFO(block_manager_config_.reserved_device_memory_ratio > 0.0,
                       "reserved_device_memory_ratio must be large than 0.0");
  KLLM_CHECK_WITH_INFO(block_manager_config_.block_host_memory_factor >= 0.0, "block_host_memory_factor should >= 0.0");

  const size_t alignment_bytes = 8;
  size_t device_block_memory_size = 0;
  if (block_manager_config_.block_device_memory_ratio >= 0.0) {
    device_block_memory_size =
        DivRoundDown(std::min((static_cast<size_t>(device_total * block_manager_config_.block_device_memory_ratio)),
                              device_free),
                     alignment_bytes) *
        alignment_bytes;
  } else {
    size_t reserved_memory_size = 0;
    if (!DeviceMemoryPool::Empty()) {
      reserved_memory_size =
          DivRoundUp(DynamicMemoryCounter::GetMemoryBytes(0) * block_manager_config_.dynamic_reusable_memory_ratio,
                     alignment_bytes) *
          alignment_bytes;
    } else {
      reserved_memory_size = DivRoundUp((device_total * device_reserved_ratio), alignment_bytes) * alignment_bytes;
    }

    device_block_memory_size =
        DivRoundDown((reserved_memory_size < device_free ? device_free - reserved_memory_size : 0ul), alignment_bytes) *
        alignment_bytes;
  }

  const float block_host_memory_ratio = 0.8;
  size_t host_block_memory_size =
      DivRoundDown(
          static_cast<size_t>(std::min(device_block_memory_size * block_manager_config_.block_host_memory_factor,
                                       host_free * block_host_memory_ratio)),
          alignment_bytes) *
      alignment_bytes;

  KLLM_LOG_INFO << "Get block memory info, host_free:" << host_block_memory_size
                << ", device_free:" << device_block_memory_size
                << ", block_size:" << block_manager_config_.host_allocator_config.block_size;

  size_t device_blocks_num = device_block_memory_size / block_manager_config_.device_allocator_config.block_size;
  size_t host_blocks_num = host_block_memory_size / block_manager_config_.host_allocator_config.block_size;
  KLLM_LOG_INFO << "Device blocks limit = " << device_blocks_num << "."
                << "Host blocks limit = " << host_blocks_num << ".";
  // Control max device_blocks_num through KLLM_MAX_DEVICE_BLOCKS
  const char *max_blocks_str = std::getenv("KLLM_MAX_DEVICE_BLOCKS");
  if (max_blocks_str != nullptr) {
    try {
      size_t max_device_blocks = std::stoull(max_blocks_str);
      if (max_device_blocks >= 1 && max_device_blocks <= device_blocks_num) {
        device_blocks_num = max_device_blocks;
        KLLM_LOG_INFO << "Using custom max device blocks limit: " << max_device_blocks;
      }
    } catch (const std::exception &e) {
    }
  }
  KLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_blocks_num;

  block_manager_config_.device_allocator_config.blocks_num = device_blocks_num;
  block_manager_config_.host_allocator_config.blocks_num = host_blocks_num;

  return Status();
}

Status ScheduleConfigParser::ResetPipelineBlockNumber() {
  // Get block number from pipeline config if in distributed mode.
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);

  size_t device_blocks_num = pipeline_config.device_block_num;
  size_t host_block_num = pipeline_config.host_block_num;

  KLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_block_num;

  block_manager_config_.device_allocator_config.blocks_num = device_blocks_num;
  block_manager_config_.host_allocator_config.blocks_num = host_block_num;

  return Status();
}

size_t ScheduleConfigParser::GetTotalDeviceBlockNum() {
  return block_manager_config_.device_allocator_config.blocks_num;
}

size_t ScheduleConfigParser::GetTotalHostBlockNum() { return block_manager_config_.host_allocator_config.blocks_num; }

std::vector<int> ScheduleConfigParser::GetDataParaGroupDevices(int dp_id) {
  size_t device_count = runtime_config_.parallel_basic_config.tensor_parallel_size;
  size_t group_device_count = device_count / runtime_config_.parallel_basic_config.attn_data_parallel_size;

  std::vector<int> group_devices;
  for (size_t i = 0; i < group_device_count; ++i) {
    group_devices.push_back(dp_id * group_device_count + i);
  }

  return group_devices;
}

bool ScheduleConfigParser::IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

size_t ScheduleConfigParser::GetTransferLayerChunkSize() { return batch_scheduler_config_.transfer_layer_chunk_size; }

void ScheduleConfigParser::InitializeExpertParallelConfig() {
  const char *expert_master_host = std::getenv("EXPERT_MASTER_HOST");
  const char *expert_master_port = std::getenv("EXPERT_MASTER_PORT");
  const char *expert_node_rank = std::getenv("EXPERT_NODE_RANK");
  const char *use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");

  ExpertParallelConfig expert_parallel_config;
  GetExpertParallelConfig(expert_parallel_config);
  expert_parallel_config.expert_node_rank = expert_node_rank ? std::stoi(expert_node_rank) : 0;
  expert_parallel_config.expert_para_size = runtime_config_.parallel_basic_config.expert_parallel_size;
  expert_parallel_config.expert_tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size /
                                                   runtime_config_.parallel_basic_config.expert_parallel_size;
  expert_parallel_config.global_expert_para_size =
      expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  if (expert_parallel_config.expert_world_size > 1) {
    if (!expert_master_host || !expert_master_port) {
      throw std::runtime_error(
          "The environment variable MASTER_HOST and MASTER_PORT must be set in distributed expert parallel mode.");
    }
  }

  expert_parallel_config.expert_master_host = expert_master_host ? expert_master_host : "";
  expert_parallel_config.expert_master_port = expert_master_port ? std::stoi(expert_master_port) : 0;

  if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) expert_parallel_config_.use_tcp = true;

  KLLM_LOG_INFO << "InferenceServer initialize expert parallel config, expert_master_host:"
                << expert_parallel_config.expert_master_host
                << ", expert_master_port:" << expert_parallel_config.expert_master_port
                << ", expert_world_size:" << expert_parallel_config.expert_world_size
                << ", expert_para_size:" << expert_parallel_config.expert_para_size
                << ", gloal_expert_para_size:" << expert_parallel_config.global_expert_para_size
                << ", expert_node_rank:" << expert_parallel_config.expert_node_rank
                << ", use_tcp: " << expert_parallel_config.use_tcp;
  SetExpertParallelConfig(expert_parallel_config);
}

}  // namespace ksana_llm
