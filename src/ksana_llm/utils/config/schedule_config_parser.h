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
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace ksana_llm {

enum PreemptMode { SWAP = 0, RECOMPUTE = 1 };

enum ScheduleStrategy { CONTINUOUS_BATCHING = 0 };

enum PPMultibatchWBStrategy { NO_WB = 0, NO_DYNAMIC_WB = 1, WB_BATCH_REQ = 2, WB_BATCH_TOKEN = 3, WB_REQ_TOKEN = 4 };

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
  size_t max_step_token_num = 4096;  // to be removed

  // The max batch size.
  size_t max_batch_size = 8;  // to be removed

  size_t max_pp_batch_num = 1;  // to be removed

  // The max vocab size.
  size_t max_vocab_size = 32000;  // TODO(robertyuan): Use model_config.vocab_size. To be removed

  // The maximum length the generated tokens can have
  // orresponds to the length of the input prompt + max_new_tokens.
  size_t max_token_len = 2048;  // to be removed

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

  // The max batch size for pre-transfer operations.
  size_t max_pretransfer_batch_size = 64;

  // The number of layers to pack together for each transfer operation (chunk transfer).
  size_t transfer_layer_chunk_size = 1;

  bool enable_speculative_decoding = false;
  bool enable_mtp_module = false;
  bool enable_async = false;

  bool enable_xgrammar = false;
};

struct AllocatorConfig {
  // The preallocated blocks.
  size_t blocks_num = 0;

  // The block size, in bytes.
  size_t block_size = 0;

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
  bool enable_block_checksum = false;

  // The ratio of reserved device memory.
  float reserved_device_memory_ratio = 0.05;

  // The ratio of block device memory. use all left memory if less than 0.0.
  float block_device_memory_ratio = -1.0;

  // The scale fator of block host memory.
  float block_host_memory_factor = 10.0;

  // The ratio of dynamic reusable memory.
  float dynamic_reusable_memory_ratio = 1.0;
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
  bool enable_blocked_multi_token_forwarding_kv = false;
  DataType kv_cache_dtype;    // kv_cache storage type
  size_t block_token_num{0};  // The max token number of one block.
  size_t block_size{0};       // The block size, in bytes.

  // User preference for FlashAttention implementation selection.
  enum class FlashAttnImplChoice {
    AUTO = 0,  // Auto-detect by hardware and availability (default)
    FA3,       // FlashAttention 3
    VLLM_V26,  // vLLM FlashAttention 2.6+
    FA2_V26,   // FlashAttention 2.6+
    FA2_V25    // FlashAttention 2.5+
  };
  FlashAttnImplChoice flash_attn_impl_choice = FlashAttnImplChoice::AUTO;

  // std::vector<float> k_scales;  // to be removed
  // std::vector<float> v_scales;  // to be removed
};

struct ParallelismBasicConfig {
  size_t tensor_parallel_size{1};
  size_t attn_data_parallel_size{1};
  size_t attn_tensor_parallel_size{1};  // Determined by tp/dp
  size_t expert_parallel_size{1};
  size_t expert_world_size{1};
  size_t moe_tensor_para_size{1};
};

enum W4AFP8_MOE_BACKEND { Default = 0, GroupTriton = 1, TensorTriton = 2 };

// Config info used during runtime
// Some configs are determined by ModelConfig and BatchSchedulerConfig
struct RuntimeConfig {
  // Group 1: parallelism config
  ParallelismBasicConfig parallel_basic_config;

  // Group 2: execution graph config
  // For attention backend.
  AttnBackendConfig attn_backend_config;
  bool enable_full_shared_expert = false;
  bool separate_prefill_decode = false;

  bool enable_prefix_caching = false;  // Whether enable prefix caching.
  bool enable_flexible_caching = false;

  // Whether to dump eplb data.
  bool enable_dump_eplb_data = false;
  bool enable_load_eplb_weight = false;

  // Backend type of w4afp8 moe
  W4AFP8_MOE_BACKEND w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;

  // Whether to normalize q and k before rotary position embedding in attention.
  // bool enable_qk_pre_norm_before_rotary_pos = false;

  // Schedule related. determined by schedule configs and cache related configs.
  size_t max_pp_batch_num{1};  // max number of batchs in pipeline parallel.
  int max_batch_size;
  size_t max_seq_len;         // The max token number of a sequence
  size_t max_step_token_num;  //  The max token number of step

  bool enable_mtp_module = false;
  bool enable_speculative_decoding = false;
  bool enable_async = false;
  // DataType kv_cache_dtype;

  // TODO(robertyuan): No body set it?
  bool embed_tokens_use_cpu{false};  // Embed_tokens gather operation is processed on the CPU.

  // data type of intermediate data: input data and output data type of kernels
  DataType inter_data_type;

  bool enable_o_proj_out_of_dp = false;  // Whether to enable out-of-data-parallelism for o_proj in attention.

  bool is_profile_mode = false;  // Only used for profiling performance
};

class ScheduleConfigParser {
 public:
  ScheduleConfigParser();

  // Parse environment from YAML reader.
  Status ParseScheduleConfig(YamlReader &yaml_reader, ModelConfig &model_config);

  void Reset();

  Status UpdateModelConfig(ModelConfig &model_config);

  void UpdateMembers(const std::string &model_dir, ModelConfig &model_config, std::string &kv_cache_dtype_str);

  Status CheckEnvironment();

  // Get the config of batch manager.
  Status GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config);
  void SetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config);

  // Get the config of cached manager.
  Status GetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  void SetCacheManagerConfig(CacheManagerConfig &cache_manager_config);

  Status GetRuntimeConfig(RuntimeConfig &runtime_config);

  // Whether the auto-prefix-caching is enabled.
  bool IsPrefixCachingEnabled();

  size_t GetTransferLayerChunkSize();

  // Get the config of block manager.
  Status GetBlockManagerConfig(BlockManagerConfig &block_manager_config);

  // TODO(yancyliu): remove from here later.
  void SetBlockManagerConfig(const BlockManagerConfig &block_manager_config);
  Status CalculateBlockNumber();
  Status ResetPipelineBlockNumber();
  size_t GetTotalDeviceBlockNum();
  size_t GetTotalHostBlockNum();
  bool IsEnableBlockChecksum();
  std::vector<int> GetDataParaGroupDevices(int dp_id);

  void SetTensorParallelSize(size_t tensor_parallel_size) {
    runtime_config_.parallel_basic_config.tensor_parallel_size = tensor_parallel_size;
  }

  void SetAttnDataParallelSize(size_t attn_data_parallel_size) {
    runtime_config_.parallel_basic_config.attn_data_parallel_size = attn_data_parallel_size;
  }

  void SetExpertParallelSize(size_t expert_parallel_size) {
    runtime_config_.parallel_basic_config.expert_parallel_size = expert_parallel_size;
  }

  size_t GetMaxBatchSize() const { return batch_scheduler_config_.max_batch_size; }

  // Modify reserved_device_memory_ratio
  void SetReservedDeviceRatio(float reserved_device_memory_ratio);

  // Set and get multiple node pipeline config.
  void SetPipelineConfig(const PipelineConfig &pipeline_config) { pipeline_config_ = pipeline_config; }

  Status GetPipelineConfig(PipelineConfig &pipeline_config) const {
    pipeline_config = pipeline_config_;
    return Status();
  }

  void GetAttnBackendConfig(AttnBackendConfig &attn_backend_config) {
    attn_backend_config = runtime_config_.attn_backend_config;
  }

  void SetAttnBackendConfig(const AttnBackendConfig &attn_backend_config) {
    runtime_config_.attn_backend_config = attn_backend_config;
  }

  Status GetExpertParallelConfig(ExpertParallelConfig &expert_parallel_config) const {
    expert_parallel_config = expert_parallel_config_;
    return Status();
  }

  void SetExpertParallelConfig(const ExpertParallelConfig &expert_parallel_config) {
    expert_parallel_config_ = expert_parallel_config;
  }

  Status GetConnectorConfigs(ConnectorConfig &connector_config) const {
    // 检查connector_config_是否已初始化 - group_role
    if (connector_config_.group_role == GroupRole::NONE || connector_config_.router_endpoint.empty()) {
      return Status(RET_CONFIG_NOT_FOUND, "Connector config is not initialized.");
    }
    connector_config = connector_config_;
    return Status();
  }

  void SetConnectorConfigs(const ConnectorConfig &connector_config) {
    connector_config_ = connector_config;
    return;
  }

  // Init disaggregating prefill and decode connector config
  void InitConnectorConfig(YamlReader &yaml_reader);

  // Calculate block size via model configs.
  Status InitializeBlockManagerConfig(const ModelConfig &model_config);

  // Init Expert-Parallel Config from env.
  void InitializeExpertParallelConfig();

  // Get block size in bytes.
  size_t GetCacheBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                           const BlockManagerConfig &block_manager_config);

 private:
  size_t GetCommonBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                            const BlockManagerConfig &block_manager_config);

  size_t GetDeepSeekV3BlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
                                const BlockManagerConfig &block_manager_config);

 private:
  // The config of batch schedule.
  BatchSchedulerConfig batch_scheduler_config_;

  // The config used by cache manager.
  CacheManagerConfig cache_manager_config_;

  // The config of block manager.
  BlockManagerConfig block_manager_config_;

  RuntimeConfig runtime_config_;

  // TODO(robertyuan): This two configs will be set by data channel, fix them later
  // For distributed multiple node pipeline.
  PipelineConfig pipeline_config_;
  // For expert parallel.
  ExpertParallelConfig expert_parallel_config_;

  // Store parsed connector configurations
  ConnectorConfig connector_config_;
};

}  // namespace ksana_llm
