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
  MOE_QUANT_GPTQ
};

__attribute__((unused)) static std::string GetQuantModeString(QuantMode quant_mode) {
  static const std::unordered_map<QuantMode, std::string> quant_mode_map{
      {QUANT_NONE, "None"},
      {QUANT_GPTQ, "GPTQ"},
      {QUANT_AWQ, "AWQ"},
      {QUANT_FP8_E4M3, "FP8_E4M3"},
      {QUANT_BLOCK_FP8_E4M3, "BLOCK_FP8_E4M3"},
      {MOE_QUANT_NONE, "MOE_QUANT None"},
      {MOE_QUANT_BLOCK_FP8_E4M3, "MOE_QUANT BLOCK_FP8_E4M3"},
      {MOE_QUANT_FP8_E4M3, "MOE_QUANT FP8_E4M3"},
      {MOE_QUANT_GPTQ, "MOE_QUANT GPTQ"}};
  return quant_mode_map.count(quant_mode) ? quant_mode_map.at(quant_mode) : "Unknown";
}

/*
 * LinearComputeBackend:
 * - DEFAULT_LINEAR_BACKEND:
 *     The default linear backend. Suitable for FP8, FP16, BF16 and FP32 models.
 *     Internally may use cuBLAS, CUTLASS or other implementations.
 * - CUTLASS_LINEAR_BACKEND:
 *     An int4-only backend. It is the default choice for int4 inference and
 *     is suitable for GPTQ/AWQ models.
 * - MARLIN_LINEAR_BACKEND:
 *     An int4-only backend suitable for GPTQ/AWQ models. Additionally supports the
 *     `act_order` feature.
 * - MACHETE_LINEAR_BACKEND:
 *     An int4-only backend restricted to SM90 (NVIDIA Hopper class). Currently
 *     only used by DeepSeek GPTQ int4 model.
 */
enum LinearComputeBackend {
  DEFAULT_LINEAR_BACKEND,
  CUTLASS_LINEAR_BACKEND,
  MARLIN_LINEAR_BACKEND,
  MACHETE_LINEAR_BACKEND
};

__attribute__((unused)) static std::string GetLinearComputeBackendString(LinearComputeBackend linear_compute_backend) {
  static const std::unordered_map<LinearComputeBackend, std::string> linear_compute_backend_map{
      {DEFAULT_LINEAR_BACKEND, "default_linear"},
      {CUTLASS_LINEAR_BACKEND, "cutlass_linear"},
      {MARLIN_LINEAR_BACKEND, "marlin_linear"},
      {MACHETE_LINEAR_BACKEND, "machete_linear"}};
  return linear_compute_backend_map.count(linear_compute_backend)
             ? linear_compute_backend_map.at(linear_compute_backend)
             : "Unknown";
}

/*
 * MoeComputeBackend:
 * - DEFAULT_MOE_BACKEND:
 *     The default MoE backend based on cutlass. General-purpose implementation that supports
 *     common models such as Qwen, LLaMA, etc.
 * - MARLIN_MOE_BACKEND:
 *     An int4-only MoE backend. Intended for int4 versions of models like Qwen
 *     and LLaMA.
 * - TRITON_MOE_BACKEND:
 *     DeepSeek-only backend that implements a fused-MoE strategy (using Triton).
 *     Supports FP16, BF16, FP8-blockwise and int4.
 * - CUTLASS_MOE_BACKEND:
 *     DeepSeek-only backend that implements a group-GEMM strategy (using CUTLASS).
 *     Only supports the w4afp8 weight format.
 */
enum MoeComputeBackend { DEFAULT_MOE_BACKEND, CUTLASS_MOE_BACKEND, MARLIN_MOE_BACKEND, TRITON_MOE_BACKEND };

__attribute__((unused)) static std::string GetMoeComputeBackendString(MoeComputeBackend moe_compute_backend) {
  static const std::unordered_map<MoeComputeBackend, std::string> moe_compute_backend_map{
      {DEFAULT_MOE_BACKEND, "default_moe"},
      {CUTLASS_MOE_BACKEND, "cutlass_moe"},
      {MARLIN_MOE_BACKEND, "marlin_moe"},
      {TRITON_MOE_BACKEND, "triton_moe"}};
  return moe_compute_backend_map.count(moe_compute_backend) ? moe_compute_backend_map.at(moe_compute_backend)
                                                            : "Unknown";
}

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

  bool input_scale = false;

  LinearComputeBackend backend = DEFAULT_LINEAR_BACKEND;

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

  // Max sequence length during model training (unit: token num)
  size_t max_training_seq_len;

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
  size_t num_key_value_heads;  // nobody use it, to be removed
  int max_position_embeddings;

  std::vector<float> k_scales;  // to be removed
  std::vector<float> v_scales;  // to be removed

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

  // Whether to normalize q and k before rotary position embedding in attention.
  bool enable_qk_pre_norm_before_rotary_pos = false;  // to be removed

  bool enable_add_qkv_bias = false;  // TODO(robertyuan): this is a model config, not func switch. Change name

  // for llama4
  std::vector<size_t> no_rope_layers;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  size_t attention_chunk_size = 0;

  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Scaling the hidden states of residual connections.
  float scale_depth = 1.f;
  // For choose lm_head.linear_head weight (for bge-reranker-v2-minicpm-layerwise)
  int start_layer = 0;
};
class EnvModelConfigParser {
 public:
  EnvModelConfigParser(const std::string &weight_quant_method, const std::string &gptq_backend)
      : weight_quant_method_(weight_quant_method), gptq_backend_(gptq_backend) {}

  // Parse model config from model dir.
  Status ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir,
                          const std::string &model_config_filename, ModelConfig &model_config);

  // Parse Model Quant Config
  void ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                             std::string &yaml_weight_quant_method, std::string &yaml_gptq_backend);

 private:
  // Parse model config from GGUF file.
  Status ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config);

 private:
  // The config of quantization.
  std::string weight_quant_method_;

  // The backend of gptq/awq quantization.
  std::string gptq_backend_;
};

}  // namespace ksana_llm
