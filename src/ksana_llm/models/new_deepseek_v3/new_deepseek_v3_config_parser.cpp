/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config_parser.h"

#include <memory>

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/utils/config/quant_config_parser.h"
#include "ksana_llm/utils/json_config_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {
NewDeepSeekV3ConfigParser::NewDeepSeekV3ConfigParser() {}
NewDeepSeekV3ConfigParser::~NewDeepSeekV3ConfigParser() {}

Status NewDeepSeekV3ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                   const ParallelismBasicConfig &parallel_basic_config,
                                                   const std::string &model_dir,
                                                   std::shared_ptr<BaseModelConfig> &model_config) {
  KLLM_LOG_INFO << "Parse config using new deepseek v3 config parser" << std::endl;
  std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config = std::make_shared<NewDeepSeekV3Config>();
  model_config = new_deepseek_v3_config;

  new_deepseek_v3_config->model_dir = model_dir;

  auto env = Singleton<Environment>::GetInstance();
  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  env->GetExpertParallelConfig(new_deepseek_v3_config->expert_parallel_config);
  new_deepseek_v3_config->tensor_para_size = parallel_basic_config.tensor_parallel_size;
  new_deepseek_v3_config->attn_data_para_size = parallel_basic_config.attn_data_parallel_size;
  size_t ep = parallel_basic_config.expert_parallel_size;
  new_deepseek_v3_config->expert_para_size = ep == 0 ? 1 : ep;
  new_deepseek_v3_config->moe_tensor_para_size =
      new_deepseek_v3_config->tensor_para_size / new_deepseek_v3_config->expert_para_size;

  new_deepseek_v3_config->weight_data_type = GetModelDataType(config_json);
  // 1. Parse common config, use deepseekv2-lite as default values
  new_deepseek_v3_config->type = config_json.value("model_type", "deepseek_v3");
  new_deepseek_v3_config->head_num = config_json.value("num_attention_heads", 16);
  new_deepseek_v3_config->num_key_value_heads = config_json.value("num_key_value_heads", 16);
  new_deepseek_v3_config->inter_size = config_json.value("intermediate_size", 10944);
  new_deepseek_v3_config->vocab_size = config_json.value("vocab_size", 102400);
  new_deepseek_v3_config->num_layer = config_json.value("num_hidden_layers", 27);
  new_deepseek_v3_config->num_nextn_predict_layers = config_json.value("num_nextn_predict_layers", 0);
  new_deepseek_v3_config->hidden_units = config_json.value("hidden_size", 2048);
  new_deepseek_v3_config->rope_theta = config_json.value("rope_theta", 10000.0f);
  new_deepseek_v3_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  new_deepseek_v3_config->layernorm_eps =
      config_json.value("layer_norm_epsilon", new_deepseek_v3_config->layernorm_eps);
  new_deepseek_v3_config->start_id = config_json.value("bos_token_id", 1);

  new_deepseek_v3_config->end_ids =
      std::vector<uint32_t>{static_cast<unsigned int>(config_json.value("eos_token_id", 2))};
  new_deepseek_v3_config->pad_id = config_json.value("pad_token_id", 0);
  new_deepseek_v3_config->max_position_embeddings = config_json.value("max_position_embeddings", 163840);
  if (!config_json.contains("tie_word_embeddings")) {
    new_deepseek_v3_config->exist_tie_embeddings_param = false;
  }
  new_deepseek_v3_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  new_deepseek_v3_config->is_visual = config_json.contains("visual");

  size_t size_per_head = new_deepseek_v3_config->hidden_units / new_deepseek_v3_config->head_num;
  new_deepseek_v3_config->size_per_head = size_per_head;
  new_deepseek_v3_config->rotary_embedding = size_per_head;

  // 2. parse moe config
  new_deepseek_v3_config->moe_config.num_experts = config_json.value("n_routed_experts", 256);
  if (new_deepseek_v3_config->moe_config.num_experts > 1) {
    new_deepseek_v3_config->moe_config.use_vllm_moe = true;
    new_deepseek_v3_config->is_moe = true;
    new_deepseek_v3_config->moe_config.moe_inter_size =
        config_json.value("moe_intermediate_size", new_deepseek_v3_config->inter_size);
    new_deepseek_v3_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
    new_deepseek_v3_config->moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 3);
    // For moe group topk config
    new_deepseek_v3_config->moe_config.num_expert_group = config_json.value("n_group", 1);
    new_deepseek_v3_config->moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    new_deepseek_v3_config->moe_config.scoring_func = config_json.value("scoring_func", "sigmoid");
    new_deepseek_v3_config->moe_config.topk_method = config_json.value("topk_method", "greedy");
    new_deepseek_v3_config->moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
    new_deepseek_v3_config->moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.0f);
    new_deepseek_v3_config->moe_config.use_e_score_correction_bias =
        (new_deepseek_v3_config->moe_config.topk_method == "noaux_tc");
  }
  new_deepseek_v3_config->moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (new_deepseek_v3_config->moe_config.num_shared_experts > 0) {
    new_deepseek_v3_config->has_shared_experts = true;
    new_deepseek_v3_config->moe_config.shared_expert_inter_size =
        new_deepseek_v3_config->moe_config.num_shared_experts * new_deepseek_v3_config->moe_config.moe_inter_size;
  }
  KLLM_LOG_INFO << fmt::format(
      "Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, moe_inter_size: {}, "
      "use_e_score_correction_bias: {}",
      new_deepseek_v3_config->moe_config.num_experts, new_deepseek_v3_config->moe_config.num_shared_experts,
      new_deepseek_v3_config->moe_config.experts_topk, new_deepseek_v3_config->moe_config.moe_inter_size,
      new_deepseek_v3_config->moe_config.use_e_score_correction_bias);

  // 3. parse mla config
  new_deepseek_v3_config->use_mla = true;
  if (config_json.contains("q_lora_rank") && config_json["q_lora_rank"].is_number()) {
    new_deepseek_v3_config->mla_config.q_lora_rank = config_json.value("q_lora_rank", 1536);
  } else {
    new_deepseek_v3_config->mla_config.q_lora_rank = 0;
  }
  new_deepseek_v3_config->mla_config.kv_lora_rank = config_json.value("kv_lora_rank", 512);
  new_deepseek_v3_config->mla_config.qk_nope_head_dim = config_json.value("qk_nope_head_dim", 128);
  new_deepseek_v3_config->mla_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  new_deepseek_v3_config->mla_config.v_head_dim = config_json.value("v_head_dim", 128);
  new_deepseek_v3_config->size_per_head =
      new_deepseek_v3_config->mla_config.qk_nope_head_dim + new_deepseek_v3_config->mla_config.qk_rope_head_dim;
  KLLM_LOG_INFO << fmt::format(
      "Using mla model, q_lora_rank: {}, kv_lora_rank: {}, qk_nope_head_dim: {}, qk_rope_head_dim: {}, v_head_dim: {}",
      new_deepseek_v3_config->mla_config.q_lora_rank, new_deepseek_v3_config->mla_config.kv_lora_rank,
      new_deepseek_v3_config->mla_config.qk_nope_head_dim, new_deepseek_v3_config->mla_config.qk_rope_head_dim,
      new_deepseek_v3_config->mla_config.v_head_dim);

  // 4. parse optional deepseek sparse mla config
  if (new_deepseek_v3_config->type == "deepseek_v32") {
    new_deepseek_v3_config->use_dsa = true;
    new_deepseek_v3_config->dsa_config.index_head_dim = config_json.value("index_head_dim", 128);
    new_deepseek_v3_config->dsa_config.index_n_heads = config_json.value("index_n_heads", 64);
    new_deepseek_v3_config->dsa_config.index_topk = config_json.value("index_topk", 2048);
    KLLM_LOG_INFO << fmt::format("Using dsa model, index_head_dim: {}, index_n_heads: {}, index_topk: {}",
                                 new_deepseek_v3_config->dsa_config.index_head_dim,
                                 new_deepseek_v3_config->dsa_config.index_n_heads,
                                 new_deepseek_v3_config->dsa_config.index_topk);
  }

  // 5. parse quantization config
  ParseQuantConfig(config_json, new_deepseek_v3_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

Status NewDeepSeekV3ConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                                   std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config,
                                                   const std::string &yaml_weight_quant_method,
                                                   const std::string &yaml_gptq_backend) {
  // 解析 quantize_config.json 和 hf_quant_config.json
  nlohmann::json quantize_config = ReadJsonFromFile(new_deepseek_v3_config->model_dir + "/quantize_config.json");
  nlohmann::json hf_quant_config = ReadJsonFromFile(new_deepseek_v3_config->model_dir + "/hf_quant_config.json");

  // 解析三种不同的情况
  nlohmann::json quantization_config;
  if (config_json.contains("quantization_config")) {
    // config.json 中有 quantization_config 字段（默认情况）
    new_deepseek_v3_config->is_quant = true;
    quantization_config = config_json["quantization_config"];
  } else if (!quantize_config.empty()) {
    // config.json 中没有 quantization_config 字段，但有额外文件 quantization_config
    KLLM_LOG_INFO << "No quantization_config in config.json, but find quantize_config.json";
    new_deepseek_v3_config->is_quant = true;
    quantization_config = quantize_config;
  } else if (!hf_quant_config.empty()) {
    // config.json 中没有 quantization_config 字段，没有文件 quantization_config.json，但有 hf_quant_config.json
    KLLM_LOG_INFO << "No quantization_config in config.json, no quantize_config.json, but find hf_quant_config.json";
    new_deepseek_v3_config->is_quant = true;
    quantization_config = ParseAndConvertQuantConfig(hf_quant_config);
  }

  if (new_deepseek_v3_config->is_quant) {
    std::string quant_method = quantization_config.at("quant_method");
    if (quant_method == "gptq" || quant_method == "rtn") {
      ParseGPTQQuantConfig(quantization_config, new_deepseek_v3_config->is_moe, new_deepseek_v3_config->quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(quantization_config, new_deepseek_v3_config->is_moe, new_deepseek_v3_config->quant_config);
    } else if (quant_method == "mixed") {
      auto configs = quantization_config["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it) {
        QuantConfig quant_config;
        quant_method = quantization_config["configs"][it.key()]["method"];
        if (quant_method == "gptq" || quant_method == "rtn") {
          ParseGPTQQuantConfig(quantization_config["configs"][it.key()], new_deepseek_v3_config->is_moe, quant_config);
        } else if (quant_method == "fp8") {
          ParseFP8QuantConfig(quantization_config["configs"][it.key()], new_deepseek_v3_config->is_moe, quant_config);
        } else {
          KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
        }
        auto layer_mapping = quantization_config["layer_mapping"][it.key()];
        quant_config.pattern_layers = layer_mapping["pattern_layers"].get<std::vector<std::string>>();
        quant_config.ignored_layers = layer_mapping["ignored_layers"].get<std::vector<std::string>>();
        if (layer_mapping["default_config"]) {
          new_deepseek_v3_config->quant_config = quant_config;
        } else {
          new_deepseek_v3_config->sub_quant_configs.push_back(quant_config);
        }
      }
      if (new_deepseek_v3_config->sub_quant_configs.size() == 1 &&
          new_deepseek_v3_config->sub_quant_configs[0].method == QUANT_GPTQ &&
          new_deepseek_v3_config->sub_quant_configs[0].pattern_layers.size() == 1 &&
          new_deepseek_v3_config->sub_quant_configs[0].pattern_layers[0] == ".mlp.experts.") {
        new_deepseek_v3_config->quant_config.enable_moe_int4 = true;
      }
    } else {
      KLLM_THROW(fmt::format("Not support quant method: {}", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_gptq_backend.empty()) {
    if (new_deepseek_v3_config->is_moe) {
      KLLM_THROW(fmt::format("Not support quant_method {} for moe model.", yaml_weight_quant_method));
    }
    if (yaml_weight_quant_method == "fp8_e4m3") {
      // when quantization_config in config.json is null,
      // quant method is decided by quantization_config in yaml.
      new_deepseek_v3_config->is_quant = true;
      new_deepseek_v3_config->quant_config.method = QUANT_FP8_E4M3;
      new_deepseek_v3_config->quant_config.is_checkpoint_fp8_serialized = false;
      new_deepseek_v3_config->quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, new_deepseek_v3_config->quant_config.is_checkpoint_fp8_serialized,
          new_deepseek_v3_config->quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
    }
  }

  // Deepseek only support machete backend
  if (new_deepseek_v3_config->is_quant && new_deepseek_v3_config->ContainGptqWeights()) {
    new_deepseek_v3_config->quant_config.backend = MACHETE_LINEAR_BACKEND;
  } else {
    KLLM_LOG_INFO << "Not using any Quant Backend";
  }

  return Status();
}
}  // namespace ksana_llm
