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
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

ScheduleConfigParser::ScheduleConfigParser() { Reset(); }

size_t ScheduleConfigParser::GetCommonBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                                const BlockManagerConfig &block_manager_config) {
  const bool predict_nextn =
      static_cast<int>(pipeline_config.lower_nextn_layer_idx) >= static_cast<int>(model_config.num_layer);
  const size_t node_nextn_layer_num =
      predict_nextn ? pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1 : 0;
  const size_t node_layer_num =
      pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1 + node_nextn_layer_num;

  const size_t token_size = node_layer_num *
                            (model_config.num_key_value_heads / (tensor_parallel_size_ / attn_data_parallel_size_)) *
                            model_config.size_per_head;
  const size_t block_token_num = block_manager_config.device_allocator_config.block_token_num;
  const size_t block_dtype_size = GetTypeSize(block_manager_config.device_allocator_config.kv_cache_dtype);

  const size_t cache_block_size = token_size * block_token_num * 2 * block_dtype_size;
  KLLM_LOG_INFO << fmt::format("Init block num for key or value: ({} / {}) * ({} / {}) * {} = {}", node_layer_num, 1,
                               model_config.num_key_value_heads, (tensor_parallel_size_ / attn_data_parallel_size_),
                               model_config.size_per_head, token_size);

  KLLM_LOG_INFO << fmt::format("Init token size (bytes) of init block for both key and value: {} * {} * 2 * {} = {}",
                               token_size, block_manager_config.device_allocator_config.block_token_num,
                               GetTypeSize(block_manager_config.device_allocator_config.kv_cache_dtype),
                               cache_block_size);
  return cache_block_size;
}

std::tuple<size_t, size_t> ScheduleConfigParser::GetDeepSeekV3BlockSize(
    const ModelConfig &model_config, const PipelineConfig &pipeline_config,
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
  // The buffer size required for dequantization operations, in bytes.
  size_t convert_size = 0;
  if (block_manager_config.host_allocator_config.kv_cache_dtype == TYPE_FP8_E5M2 ||
      block_manager_config.host_allocator_config.kv_cache_dtype == TYPE_FP8_E4M3) {
    size_t kv_type_size = GetTypeSize(block_manager_config.host_allocator_config.kv_cache_dtype);
    size_t convert_type_size = GetTypeSize(model_config.weight_data_type);
    convert_size = cache_block_size / node_layer_num / kv_type_size * convert_type_size;
    KLLM_LOG_INFO << fmt::format("Init convert size for fp8_e5m2 or fp8_e4m3: {} / {} / {} * {} = {}", cache_block_size,
                                 node_layer_num, kv_type_size, convert_type_size, convert_size);
  }
  KLLM_LOG_INFO << fmt::format(
      "Init cache block size, node_layer_num:{}, kv_lora_rank:{}, qk_rope_head_dim:{}, block_token_num:{}, "
      "cache_block_size:{}, convert_size:{}.",
      node_layer_num, model_config.mla_config.kv_lora_rank, model_config.mla_config.qk_rope_head_dim, block_token_num,
      cache_block_size, convert_size);

  return {cache_block_size, convert_size};
}

std::tuple<size_t, size_t> ScheduleConfigParser::GetCacheBlockSize(const ModelConfig &model_config,
                                                                   const PipelineConfig &pipeline_config,
                                                                   const BlockManagerConfig &block_manager_config) {
  if (model_config.type == "deepseek_v2" || model_config.type == "deepseek_v3") {
    if (IsAbsorbWeightsEnabled()) {
      return GetDeepSeekV3BlockSize(model_config, pipeline_config, block_manager_config);
    }
    return {GetCommonBlockSize(model_config, pipeline_config, block_manager_config), 0};
  }

  return {GetCommonBlockSize(model_config, pipeline_config, block_manager_config), 0};
}

void ScheduleConfigParser::Reset() {
  tensor_parallel_size_ = 1;
  attn_data_parallel_size_ = 1;
  pipeline_parallel_size_ = 1;
  expert_parallel_size_ = 1;
  expert_world_size_ = 1;
  batch_scheduler_config_ = {};
  cache_manager_config_ = {};
  block_manager_config_ = {};
  pipeline_config_ = {};
  dp_group_status_.resize(16, false);
  expert_parallel_config_ = {};
  connector_config_ = {};
  enable_flash_mla_ = false;
  enable_full_shared_expert_ = false;
}

Status ScheduleConfigParser::ParseConfig(YamlReader &yaml_reader) {
  // Read global setting.
  tensor_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.tensor_para_size", 0);
  attn_data_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.attn_data_para_size", 1);
  pipeline_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.pipeline_para_size", 1);
  expert_world_size_ = yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  expert_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);
  enable_full_shared_expert_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_full_shared_expert", false);
  if (tensor_parallel_size_ == 0) {
    int device_size = -1;
    GetDeviceCount(&device_size);
    tensor_parallel_size_ = static_cast<size_t>(device_size);
  }

  if (attn_data_parallel_size_ > 1 && std::getenv("PYTORCH_CUDA_ALLOC_CONF") != nullptr) {
    KLLM_THROW(
        fmt::format("Set env PYTORCH_CUDA_ALLOC_CONF to backend:cudaMallocAsync while attn_data_parallel_size_ > 1 may "
                    "cause libtorch blocked, please unset it."));
  }

  KLLM_CHECK_WITH_INFO(
      tensor_parallel_size_ >= attn_data_parallel_size_,
      fmt::format("Tensor Para Size(tensor_para_size) {} should >= Attention Data Para Size(attn_data_para_size) {}",
                  tensor_parallel_size_, attn_data_parallel_size_));

  KLLM_CHECK_WITH_INFO(
      tensor_parallel_size_ % attn_data_parallel_size_ == 0,
      fmt::format("Tensor Para Size(tensor_para_size) {} % Attention Data Para Size(attn_data_para_size) {} != 0",
                  tensor_parallel_size_, attn_data_parallel_size_));

#if (defined(ENABLE_ACL) || defined(ENABLE_TOPS))
  if (attn_data_parallel_size_ > 1) {
    KLLM_THROW(
        fmt::format("Huawei Ascend does not support data parallelism, please set attn_data_parallel_size to 1."));
  }
#endif

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

  if (!(tensor_parallel_size_ > 0 && attn_data_parallel_size_ > 0)) {
    KLLM_THROW(fmt::format("Tensor Para Size {}, Data Para Size {} should > 0", tensor_parallel_size_,
                           attn_data_parallel_size_));
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

  // When MTP is enabled, each request requires calculating 2 tokens while decoding.
  batch_scheduler_config_.max_decode_tokens_per_req = batch_scheduler_config_.enable_mtp_module ? 2 : 1;

  if (attn_data_parallel_size_ > 1) {
    KLLM_CHECK_WITH_INFO(
        batch_scheduler_config_.max_step_token_num / attn_data_parallel_size_ >= batch_scheduler_config_.max_token_len,
        fmt::format("max_step_token_num({}) / attn_data_para_size({}) must >= max_token_len({})",
                    batch_scheduler_config_.max_step_token_num, attn_data_parallel_size_,
                    batch_scheduler_config_.max_token_len));
  }

  // Read block manager config.
  block_manager_config_.host_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.device_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);

  const char *enable_flash_mla = std::getenv("ENABLE_FLASH_MLA");
  if (enable_flash_mla != nullptr && strcmp(enable_flash_mla, "1") == 0) {
    enable_flash_mla_ = true;
    block_manager_config_.host_allocator_config.block_token_num = 64;
    block_manager_config_.device_allocator_config.block_token_num = 64;
    KLLM_LOG_INFO << "ENABLE_FLASH_MLA=1 detected, setting block_token_num to 64 for flash_mla";
  }
  block_manager_config_.reserved_device_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.reserved_device_memory_ratio", 0.01);
  block_manager_config_.block_device_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_device_memory_ratio", -1.0);
  block_manager_config_.block_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_host_memory_factor", 2.0);

  // Load cache manager config
  cache_manager_config_.swap_threadpool_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swap_threadpool_size", 2);
  cache_manager_config_.min_flexible_cache_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.min_flexible_cache_num", 0);
  cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  cache_manager_config_.tensor_para_size = tensor_parallel_size_;
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

  InitConnectorConfig(yaml_reader);
  return Status();
}

void ScheduleConfigParser::UpdateMembers(const ModelConfig &model_config, DataType kv_cache_dtype) {
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

  if (tensor_parallel_size_ > model_config.num_key_value_heads ||
      model_config.num_key_value_heads % tensor_parallel_size_ != 0) {
    KLLM_THROW(
        fmt::format("The size of key_value_heads cannot be evenly divided by the size of tensor_parallel_size_. "
                    "{} % {} != 0 ",
                    model_config.num_key_value_heads, tensor_parallel_size_));
  }

  if (tensor_parallel_size_ < expert_parallel_size_ || tensor_parallel_size_ % expert_parallel_size_ != 0) {
    KLLM_THROW(
        fmt::format("The size of tensor_parallel_size_ cannot be evenly divided by the size of expert_parallel_size_. "
                    "{} % {} != 0 ",
                    tensor_parallel_size_, expert_parallel_size_));
  }

  if (batch_scheduler_config_.max_token_len > 0) {
    if (batch_scheduler_config_.max_token_len > model_config.max_token_num) {
      KLLM_LOG_WARNING << fmt::format(
          "The max_token_num configured in the model's config.json is less than the "
          "max_token_len configured in the ksana yaml file. {} < {}, use {}",
          model_config.max_token_num, batch_scheduler_config_.max_token_len, model_config.max_token_num);
    } else {
      model_config.max_token_num = batch_scheduler_config_.max_token_len;
    }
  }
  batch_scheduler_config_.max_token_len = model_config.max_token_num;
  if ((batch_scheduler_config_.split_fuse_token_num == 0) &&
      (batch_scheduler_config_.max_step_token_num < model_config.max_token_num)) {
    // if no split fuse, request cannot be processed if max_step_num < input token num
    batch_scheduler_config_.max_step_token_num = model_config.max_token_num;
  }

  // TODO(robertyuan): These members should be moved to other configs
  model_config.tensor_para_size = tensor_parallel_size_;
  model_config.attn_data_para_size = attn_data_parallel_size_;
  model_config.expert_world_size = expert_world_size_;
  model_config.expert_para_size = expert_parallel_size_;
  model_config.moe_tensor_para_size = tensor_parallel_size_ / expert_parallel_size_;
  model_config.enable_full_shared_expert = enable_full_shared_expert_;

  model_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  model_config.max_batch_size = batch_scheduler_config_.max_batch_size;
  model_config.max_pp_batch_num = batch_scheduler_config_.max_pp_batch_num;
  model_config.max_step_token_num = batch_scheduler_config_.max_step_token_num;

  model_config.enable_prefix_caching = cache_manager_config_.enable_prefix_caching;

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
  auto [cache_block_size, convert_size] = GetCacheBlockSize(model_config, pipeline_config_, block_manager_config_);

  block_manager_config_.host_allocator_config.block_size = cache_block_size;
  block_manager_config_.device_allocator_config.block_size = cache_block_size;
  block_manager_config_.host_allocator_config.convert_size = 0;
  block_manager_config_.device_allocator_config.convert_size = convert_size;

  block_manager_config_.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;

  // The default block number, will be overwrited through memory usage.
  block_manager_config_.host_allocator_config.blocks_num = 512 * 10;
  block_manager_config_.device_allocator_config.blocks_num = 512;

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

Status ScheduleConfigParser::CalculateBlockNumber() {
  size_t host_total, host_free;
  size_t device_total, device_free;

  // Allocate blocks according to the memory status of device 0.
  SetDevice(0);
  Status status =
      GetDeviceMemoryInfo(block_manager_config_.device_allocator_config.device, &device_free, &device_total);
  if (!status.OK()) {
    return status;
  }

  status = GetHostMemoryInfo(&host_free, &host_total);
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
    size_t reserved_memory_size =
        DivRoundUp((device_total * block_manager_config_.reserved_device_memory_ratio), alignment_bytes) *
        alignment_bytes;
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

  size_t device_blocks_num = device_block_memory_size / (block_manager_config_.device_allocator_config.block_size +
                                                         block_manager_config_.device_allocator_config.convert_size);
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

size_t ScheduleConfigParser::GetBlockTokenNum() {
  return block_manager_config_.device_allocator_config.block_token_num;
}

size_t ScheduleConfigParser::GetConvertSize() { return block_manager_config_.device_allocator_config.convert_size; }

size_t ScheduleConfigParser::GetBlockSize() { return block_manager_config_.device_allocator_config.block_size; }

size_t ScheduleConfigParser::GetTotalDeviceBlockNum() {
  return block_manager_config_.device_allocator_config.blocks_num;
}

size_t ScheduleConfigParser::GetTotalHostBlockNum() { return block_manager_config_.host_allocator_config.blocks_num; }

DataType ScheduleConfigParser::GetKVCacheType() { return block_manager_config_.device_allocator_config.kv_cache_dtype; }

std::vector<int> ScheduleConfigParser::GetDataParaGroupDevices(int dp_id) {
  size_t device_count = tensor_parallel_size_;
  size_t group_device_count = device_count / attn_data_parallel_size_;

  std::vector<int> group_devices;
  for (size_t i = 0; i < group_device_count; ++i) {
    group_devices.push_back(dp_id * group_device_count + i);
  }

  return group_devices;
}

size_t ScheduleConfigParser::GetAttentionTensorParallel() { return tensor_parallel_size_ / attn_data_parallel_size_; }

void ScheduleConfigParser::SetDataParaGroupStatus(int dp_group_id, bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  dp_group_status_[dp_group_id] = enabled;
}

bool ScheduleConfigParser::GetDataParaGroupStatus(int dp_group_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return dp_group_status_[dp_group_id];
}

bool ScheduleConfigParser::IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

bool ScheduleConfigParser::IsFlexibleCachingEnabled() { return cache_manager_config_.min_flexible_cache_num > 0; }

bool ScheduleConfigParser::IsSpeculativeDecodingEnabled() {
  return batch_scheduler_config_.enable_speculative_decoding;
}

bool ScheduleConfigParser::IsPrefillDecodeSeparation() { return connector_config_.group_role != GroupRole::NONE; }

bool ScheduleConfigParser::IsMTPEnabled() { return batch_scheduler_config_.enable_mtp_module; }

size_t ScheduleConfigParser::GetTransferLayerChunkSize() { return batch_scheduler_config_.transfer_layer_chunk_size; }

void ScheduleConfigParser::InitializeExpertParallelConfig() {
  const char *expert_master_host = std::getenv("EXPERT_MASTER_HOST");
  const char *expert_master_port = std::getenv("EXPERT_MASTER_PORT");
  const char *expert_node_rank = std::getenv("EXPERT_NODE_RANK");
  const char *use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");

  ExpertParallelConfig expert_parallel_config;
  GetExpertParallelConfig(expert_parallel_config);
  expert_parallel_config.expert_node_rank = expert_node_rank ? std::stoi(expert_node_rank) : 0;
  expert_parallel_config.expert_para_size = expert_parallel_size_;
  expert_parallel_config.expert_tensor_para_size = tensor_parallel_size_ / expert_parallel_size_;
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
