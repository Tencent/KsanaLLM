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

#include "nlohmann/json.hpp"

#include "ksana_llm/connector/config.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace ksana_llm {
constexpr size_t kStepGenerateTokenNum = 1;  // The number of tokens that the model generated at each step

struct RoPEScalingFactor {
  std::string type{"default"};
  float factor{1.0f};
  float low_freq_factor{1.0f};
  float high_freq_factor{4.0f};
  int original_max_position_embeddings{8192};
  bool has_alpha{false};
  float scaling_alpha{1.0f};         // for dynamic alpha rope
  std::vector<int> mrope_section{};  // for multimodal rope

  // deepseek-yarn config params
  bool use_deepseek_yarn{false};
  float beta_fast{32.0f};
  float beta_slow{1.0f};
  float mscale{1.0f};
  float mscale_all_dim{1.0f};
};

enum QuantMode {
  QUANT_NONE,
  QUANT_GPTQ,
  QUANT_AWQ,
  QUANT_FP8_E4M3,
  QUANT_BLOCK_FP8_E4M3,
  MOE_QUANT_NONE,
  MOE_QUANT_BLOCK_FP8_E4M3,
  MOE_QUANT_FP8_E4M3,
  MOE_QUANT_GTPQ
};

enum GroupQuantBackend { NONE_QUANT, CUTLASS_BACKEND, MARLIN_BACKEND, MACHETE_BACKEND };

enum class DistributedCommunicationType {
  DEFAULT = 0,   // send and recevie
  SCATTER = 1,   // scatter
  ALLTOALL = 2,  // all to all
};

// The Quant informations.
struct QuantConfig {
  // The quant method
  QuantMode method = QUANT_NONE;

  // (gptq/awq) The quant bits
  size_t bits = 4;

  // (gptq/awq) The quant group size
  size_t group_size = 128;

  // (fp8-blockwise) The quant block size
  std::vector<size_t> weight_block_size;

  // (gptq) The desc act mode
  bool desc_act = false;

  GroupQuantBackend backend = NONE_QUANT;

  // (fp8) Whether weight_scale shape is empty.
  bool is_fp8_blockwise = false;

  // (fp8) Whether weight is quantized in checkpoint.
  bool is_checkpoint_fp8_serialized = false;

  // (fp8) Whether input_scale is in checkpoint.
  bool is_activation_scheme_static = false;

  // (fp8) Whether enable int4 moe in fp8 model.
  bool enable_moe_int4 = false;

  // Adaptation layers
  std::vector<std::string> pattern_layers;

  // Ignored layers
  std::vector<std::string> ignored_layers;
};

// The Moe informations.
struct MoeConfig {
  size_t num_experts{1};
  size_t num_shared_experts{0};

  size_t experts_topk;

  size_t moe_inter_size;
  size_t first_k_dense_replace = 0;
  size_t shared_expert_inter_size = 0;

  // For group topk
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  bool use_e_score_correction_bias = false;
  float routed_scaling_factor = 1.0f;

  // TODO(winminkong): 增加临时算子选择项，后期改为配置项
  bool use_vllm_moe = false;

  // for llama4
  size_t interleave_moe_layer_step = 1;
  // when is_moe == true,
  // if moe_layers.empty() means all layers' mlp use moe,
  // else only layer_idx in moe_layers use moe.
  std::vector<size_t> moe_layers;
  bool output_router_logits = true;
  float router_aux_loss_coef = 0;
  float router_jitter_noise = 0;
  bool apply_weight = false;
};

// The MLA informations.
struct MlaConfig {
  uint32_t q_lora_rank = 0;
  uint32_t kv_lora_rank = 0;

  uint32_t qk_nope_head_dim = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t v_head_dim = 0;
};

// The model informations.
struct ModelConfig {
  // The model name.
  std::string name = "";

  // The model type, such as llama.
  std::string type;

  // The dir path.
  std::string path;

  std::string tokenizer_path;

  // Type of weight
  DataType weight_data_type;

  // The max input token number of request(input)
  size_t max_token_num;

  // The max token number of step
  size_t max_step_token_num;

  size_t tensor_para_size;
  size_t attn_data_para_size;
  size_t max_pp_batch_num = 1;  // max number of batchs in pipeline parallel.

  // The expert parallel size
  size_t expert_world_size;
  size_t expert_para_size;
  size_t moe_tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t num_nextn_predict_layers = 0;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;
  std::vector<float> k_scales;
  std::vector<float> v_scales;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;
  std::vector<QuantConfig> sub_quant_configs;

  // Determines if the model is a moe model.
  bool is_moe = false;
  bool has_shared_experts = false;
  bool enable_full_shared_expert = false;
  MoeConfig moe_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  ModelFileFormat model_file_format;

  bool load_bias = true;     // Check if load all weights bias.
  int cla_share_factor = 0;  // Determines the number of layers that share k and v.
  bool use_cla = false;
  bool use_qk_norm = false;  // Check if normlize the attention out q and k.
  bool mlp_bias = false;     // Check if use bias in mlp layer.
  // For mla model
  bool use_mla = false;
  MlaConfig mla_config;

  // Whether enable prefix caching.
  bool enable_prefix_caching = false;

  // Whether to normalize q and k before rotary position embedding in attention.
  bool enable_qk_pre_norm_before_rotary_pos = false;

  bool enable_add_qkv_bias = false;

  // for llama4
  std::vector<size_t> no_rope_layers;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  size_t attention_chunk_size = 0;
};

enum PreemptMode { SWAP = 0, RECOMPUTE = 1 };

enum ScheduleStrategy { CONTINUOUS_BATCHING = 0 };

enum PPMultibatchWBStrategy { NO_WB = 0, NO_DYNAMIC_WB = 1, WB_BATCH_REQ = 2, WB_BATCH_TOKEN = 3 };

struct BatchSchedulerConfig {
  // The batch schedule strategy.
  ScheduleStrategy schedule_strategy = ScheduleStrategy::CONTINUOUS_BATCHING;

  // PP Multibatch workload balance strategy
  PPMultibatchWBStrategy pp_multibatch_wb_strategy = PPMultibatchWBStrategy::NO_WB;

  // Max waiting time in millisecond.
  size_t waiting_timeout_in_ms = 600000;

  // The max queue len of waiting request.
  size_t max_waiting_queue_len = 256;

  // The max token number for one scheduler step.
  size_t max_step_token_num = 4096;

  // The max batch size.
  size_t max_batch_size = 8;

  size_t max_pp_batch_num = 1;  // max number of batchs in pipeline parallel.

  // The max vocab size.
  size_t max_vocab_size = 32000;

  // The maximum length the generated tokens can have
  // orresponds to the length of the input prompt + max_new_tokens.
  size_t max_token_len = 2048;

  // The swapin block threshold.
  float swapout_block_threshold = 1.0;

  // The swapout block threshold.
  float swapin_block_threshold = 2.0;

  // The launch block threshold.
  float launch_block_threshold = 2.0;

  // The preempt mode in case of insufficient GPU blocks.
  PreemptMode preempt_mode = SWAP;

  // This parameter controls the maximum number of tokens processed in a single
  // inference round. Setting it to 256 means that during inference, each
  // processing step (or "split") will handle up to 256 tokens. If set to 0, it
  // indicates that there is no limit on the number of tokens processed, and the
  // model will attempt to process the entire input at once. Adjusting this
  // parameter can help balance inference speed and resource consumption,
  // especially when dealing with long texts.
  size_t split_fuse_token_num = 0;

  // The number of tokens per request requiring decode computation defaults to 1. When MTP or speculative decoding is
  // enabled, a single request will compute multiple tokens.
  size_t max_decode_tokens_per_req = 1;

  bool enable_speculative_decoding = false;
  bool enable_mtp_module = false;
};

struct AllocatorConfig {
  // The preallocated blocks.
  size_t blocks_num = 0;

  // The block size, in bytes.
  size_t block_size = 0;

  // The buffer size required for dequantization operations, in bytes.
  size_t convert_size = 0;

  // kv_cache storage type
  DataType kv_cache_dtype;

  // The max token number of one block.
  size_t block_token_num;

  MemoryDevice device;
};

struct BlockManagerConfig {
  // The config of allocator for cpu/gpu/npu.
  AllocatorConfig host_allocator_config;
  AllocatorConfig device_allocator_config;

  // The ratio of reserved device memory.
  float reserved_device_memory_ratio = 0.05;

  // The ratio of block device memory. use all left memory if less than 0.0.
  float block_device_memory_ratio = -1.0;

  // The scale fator of block host memory.
  float block_host_memory_factor = 10.0;
};

// For cached manager, used for auto-prefix-caching.
struct CacheManagerConfig {
  // The token number of every block, not changed after created.
  size_t block_token_num = 16;

  // The tp num, cache manager use this to allocat blocks for every token.
  size_t tensor_para_size = 2;

  // The minimum consecutive length of flexible cache instances that can be
  // queried.
  size_t min_flexible_cache_num = 0;

  // The thread number used for async swap in/out.
  size_t swap_threadpool_size = 2;

  // Whether enable prefix caching.
  bool enable_prefix_caching = false;
};

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

// For multiple node pipeline.
struct PipelineConfig {
  std::string master_host;
  uint16_t master_port;

  // Default for standalone mode.
  size_t world_size = 1;
  size_t node_rank = 0;

  // layer id range.
  int16_t lower_layer_idx = -1;
  int16_t upper_layer_idx = -1;

  // netxn layer id range.
  int16_t lower_nextn_layer_idx = -1;
  int16_t upper_nextn_layer_idx = -1;

  // The cache block num.
  // All pipeline nodes must be same.
  size_t device_block_num;
  size_t host_block_num;

  // The current port for data transfer.
  std::string data_host;
  uint16_t data_port;

  // The downstream data port for data transfer.
  std::string downstream_host;
  uint16_t downstream_port;

  // The nccl unique_id.
  char nccl_unique_id[128];

  DistributedCommunicationType pipeline_para_comm_type = DistributedCommunicationType::DEFAULT;

  void SetDistributeRelatedConfig() {
    const char *master_host_env = std::getenv("MASTER_HOST");
    const char *master_port_env = std::getenv("MASTER_PORT");
    const char *world_size_env = std::getenv("WORLD_SIZE");
    const char *node_rank_env = std::getenv("NODE_RANK");

    world_size = world_size_env ? std::stoi(world_size_env) : 1;
    node_rank = node_rank_env ? std::stoi(node_rank_env) : 0;
    if (world_size > 1) {
      if (!master_host_env || !master_port_env) {
        throw std::runtime_error(
            "The environment variable MASTER_HOST and MASTER_PORT must be set in distributed mode.");
      }
    }

    master_host = master_host_env ? master_host_env : "";
    master_port = master_port_env ? std::stoi(master_port_env) : 0;

    KLLM_LOG_INFO << "Initialize pipeline config, master_host:" << master_host << ", master_port:" << master_port
                  << ", world_size:" << world_size << ", node_rank:" << node_rank;
  }
};

struct ExpertParallelConfig {
  // Maser node info for expert parallelism.
  std::string expert_master_host;
  uint16_t expert_master_port;

  // Default for standalone mode.
  size_t expert_world_size = 1;
  size_t expert_para_size = 1;
  size_t expert_node_rank = 0;
  // expert_tensor_para_size = tensor_para_size / expert_para_size;
  size_t expert_tensor_para_size = 1;
  // expert_global_para_size = expert_para_size * expert_world_size;
  size_t global_expert_para_size = 1;

  size_t local_num_experts = 1;

  // I.E. expert_para_size = 4, the local_expert_rank = {0, 1, 2, 3}
  // tensor_para_size = 8, expert_para_size = 4, device_id = 0,1,2,3...,7
  // local_expert_rank = device_id % expert_para_size
  // When to init?
  size_t local_expert_rank = 0;

  // Node info of every node.
  std::string data_host;
  uint16_t data_port;

  // The downstream data port for data transfer.
  std::string downstream_host;
  uint16_t downstream_port;

  // The data port for data transfer of other expert nodes.
  std::vector<std::string> expert_node_host;
  std::vector<uint16_t> expert_node_port;

  // Store <expert_id, ep_node_rank>.
  std::map<uint32_t, uint32_t> expert_route_table;
  std::vector<uint32_t> local_expert_rank_route;
  // Expert_id on the current node.
  std::vector<uint32_t> local_experts;

  bool enable_expert_para;
  bool use_tcp = false;

  // The nccl unique_id.  [node_rank][nccl_id]
#ifdef ENABLE_CUDA
  std::vector<std::array<char, sizeof(ncclUniqueId)> > nccl_unique_ids;
  char nccl_unique_id[sizeof(ncclUniqueId)];
#endif

  // Fix later @xingjinglu
  DistributedCommunicationType expert_para_comm_type = DistributedCommunicationType::DEFAULT;
};

// The config of attention backend.
struct AttnBackendConfig {
  bool enable_blocked_multi_token_forwarding_kv;
};

class Environment {
 public:
  Environment();

  // Parse environment from YAML config file.
  Status ParseConfig(const std::string &config_file, const std::string &model_dir_override = "");

  // Parse model config from model dir.
  Status ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir,
                          const std::string &model_config_filename = "config.json");

  // Parse command line options.
  Status ParseOptions(int argc, char **argv);

  // Get the model configs from env.
  Status GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs);

  // Get the model config by name.
  Status GetModelConfig(const std::string &model_name, ModelConfig &model_config);

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_manager_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  // Whether the auto-prefix-caching is enabled.
  bool IsPrefixCachingEnabled();

  // Whether the flexible caching is enabled.
  bool IsFlexibleCachingEnabled();

  bool IsSpeculativeDecodingEnabled();

  bool IsPrefillDecodeSeparation();

  bool IsMTPEnabled();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // TODO(yancyliu): remove from here later.
  void SetBlockManagerConfig(const BlockManagerConfig &block_manager_config);
  Status CalculateBlockNumber();
  Status ResetPipelineBlockNumber();
  size_t GetBlockTokenNum();
  size_t GetConvertSize();
  size_t GetBlockSize();
  size_t GetTotalDeviceBlockNum();
  size_t GetTotalHostBlockNum();
  DataType GetKVCacheType();
  std::vector<int> GetDataParaGroupDevices(int dp_id);

  // Whether specific dp group is enabled in current step.
  void SetDataParaGroupStatus(int dp_group_id, bool enabled);
  bool GetDataParaGroupStatus(int dp_group_id);

  // Get the config of endpoint.
  Status GetEndpointConfig(EndpointConfig &endpoint_config);

  // Get the config of profiler.
  Status GetProfilerConfig(ProfilerConfig &profiler_config);

  size_t GetTensorParallelSize() const { return tensor_parallel_size_; }

  void SetTensorParallelSize(size_t tensor_parallel_size) { tensor_parallel_size_ = tensor_parallel_size; }

  size_t GetAttnDataParallelSize() const { return attn_data_parallel_size_; }

  // Get each atten data parallel group size.
  // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the number of
  // attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp = 2, then each
  // attn dp group size is 2.
  size_t GetAttentionTensorParallel();

  void SetAttnDataParallelSize(size_t attn_data_parallel_size) { attn_data_parallel_size_ = attn_data_parallel_size; }

  size_t GetPipeLineParallelSize() const { return pipeline_parallel_size_; }

  size_t GetExpertParallelSize() { return expert_parallel_size_; }

  size_t GetExpertWorldSize() { return expert_world_size_; }

  const std::string &GetYamlGptqBackend() const { return yaml_gptq_backend_; }

  const std::string &GetYamlWeightQuantMethod() const { return yaml_weight_quant_method_; }

  bool EmbedTokensUseCpu() { return embed_tokens_use_cpu_; }

  bool IsReportVersion() { return is_version_report_; }

  bool IsCudagraphEnabled() { return cuda_graph_; }

  bool IsFlashMlaEnable() {
    const char *const enable_flash_mla = std::getenv("ENABLE_FLASH_MLA");
    return enable_flash_mla != nullptr && strcmp(std::getenv("ENABLE_FLASH_MLA"), "1") == 0;
  }

  // Modify reserved_device_memory_ratio
  void SetReservedDeviceRatio(float reserved_device_memory_ratio);

  // Set and get multiple node pipeline config.
  void SetPipelineConfig(const PipelineConfig &pipeline_config) { pipeline_config_ = pipeline_config; }

  Status GetPipelineConfig(PipelineConfig &pipeline_config) const {
    pipeline_config = pipeline_config_;
    return Status();
  }

  Status GetExpertParallelConfig(ExpertParallelConfig &expert_parallel_config) const {
    expert_parallel_config = expert_parallel_config_;
    return Status();
  }

  void SetExpertParallelConfig(const ExpertParallelConfig &expert_parallel_config) {
    expert_parallel_config_ = expert_parallel_config;
  }

  Status GetAttnBackendConfig(AttnBackendConfig &attn_backend_config) const {
    attn_backend_config = attn_backend_config_;
    return Status();
  }

  void SetAttnBackendConfig(const AttnBackendConfig &attn_backend_config) {
    attn_backend_config_ = attn_backend_config;
  }

  Status GetConnectorConfigs(ConnectorConfig &connector_config) const {
    // 检查connector_config_是否已初始化 - group_role
    if (connector_config_.group_role == GroupRole::NONE || connector_config_.router_endpoint.empty()) {
      return Status(RET_CONFIG_NOT_FOUND, "Connector config is not initialized.");
    }
    connector_config = connector_config_;
    return Status();
  }

  // Init disaggregating prefill and decode connector config
  void InitConnectorConfig(YamlReader &yaml_reader);

  // Calculate block size via model configs.
  Status InitializeBlockManagerConfig();

  // Init Expert-Parallel Config from env.
  void InitializeExpertParallelConfig();

  // Parse Model Quant Config
  void ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                             std::string &yaml_weight_quant_method, std::string &yaml_gptq_backend);

 private:
  // Check Whether the environment config is valid.
  Status CheckEnvironment();

  // Parse model config from GGUF file.
  Status ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config);

  // Get {block size, convert_size} in bytes.
  std::tuple<size_t, size_t> GetCacheBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                               const BlockManagerConfig &block_manager_config);

  size_t GetCommonBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                            const BlockManagerConfig &block_manager_config);

  std::tuple<size_t, size_t> GetDeepSeekV3BlockSize(const ModelConfig &model_config,
                                                    const PipelineConfig &pipeline_config,
                                                    const BlockManagerConfig &block_manager_config);

 private:
  // The model list that should be loaded.
  std::unordered_map<std::string, ModelConfig> model_configs_;

  // The config of batch schedule.
  BatchSchedulerConfig batch_scheduler_config_;

  // The config used by cache manager.
  CacheManagerConfig cache_manager_config_;

  // The config of block manager.
  BlockManagerConfig block_manager_config_;

  // The backend of gptq/awq quantization.
  std::string yaml_gptq_backend_;

  // The config of quantization.
  std::string yaml_weight_quant_method_;

  // The config of endpoint.
  EndpointConfig endpoint_config_;

  // The config of profiler.
  ProfilerConfig profiler_config_;

  // tp default value should be 1.
  size_t tensor_parallel_size_{1};
  size_t attn_data_parallel_size_{1};
  size_t pipeline_parallel_size_{1};
  size_t expert_parallel_size_{1};
  size_t expert_world_size_{1};

  // Whether lora is enabled.
  bool enable_lora_adapter_ = false;

  // Embed_tokens gather operation is processed on the CPU.
  bool embed_tokens_use_cpu_ = false;
  bool is_version_report_ = true;
  bool cuda_graph_ = false;
  bool enable_full_shared_expert_ = false;

  // For attention backend.
  AttnBackendConfig attn_backend_config_;

  // For distributed multiple node pipeline.
  PipelineConfig pipeline_config_;

  // For attn data parallel status
  std::vector<bool> dp_group_status_;

  // For expert parallel.
  ExpertParallelConfig expert_parallel_config_;

  // Store parsed connector configurations
  ConnectorConfig connector_config_;

  std::mutex mutex_;
};

}  // namespace ksana_llm
