/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config_parser.h"

#include <memory>

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/utils/json_config_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {
NewDeepSeekV3ConfigParser::NewDeepSeekV3ConfigParser() {}
NewDeepSeekV3ConfigParser::~NewDeepSeekV3ConfigParser() {}

Status NewDeepSeekV3ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                   const ParallelismBasicConfig &parallel_basic_config,
                                                   std::shared_ptr<BaseModelConfig> &model_config) {
  KLLM_LOG_INFO << "Parse config using new deepseek v3 config parser" << std::endl;
  std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config = std::make_shared<NewDeepSeekV3Config>();
  model_config = new_deepseek_v3_config;

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
    if (new_deepseek_v3_config->type == "deepseek_v3" || new_deepseek_v3_config->type == "deepseek_v2") {
      new_deepseek_v3_config->moe_config.use_vllm_moe = true;
    }
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
  KLLM_LOG_DEBUG << "new_deepseek_v3_config->moe_config.moe_inter_size "
                 << new_deepseek_v3_config->moe_config.moe_inter_size;
  new_deepseek_v3_config->moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (new_deepseek_v3_config->moe_config.num_shared_experts > 0) {
    new_deepseek_v3_config->has_shared_experts = true;
    new_deepseek_v3_config->moe_config.shared_expert_inter_size =
        new_deepseek_v3_config->moe_config.num_shared_experts * new_deepseek_v3_config->moe_config.moe_inter_size;
  }

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
      "Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, use_mla: {}, "
      "use_e_score_correction_bias: {}",
      new_deepseek_v3_config->moe_config.num_experts, new_deepseek_v3_config->moe_config.num_shared_experts,
      new_deepseek_v3_config->moe_config.experts_topk, new_deepseek_v3_config->use_mla,
      new_deepseek_v3_config->moe_config.use_e_score_correction_bias);

  // 4. parse quantization config
  ParseQuantConfig(config_json, new_deepseek_v3_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

void ParseGPTQQuantConfig(const nlohmann::json &config_json,
                          std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config, QuantConfig &quant_config) {
  quant_config.method = QUANT_GPTQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  quant_config.desc_act = config_json.at("desc_act");
  KLLM_LOG_INFO << fmt::format("using quant model, quant method gptq, bits: {}, group_size: {}, desc_act: {}",
                               quant_config.bits, quant_config.group_size, quant_config.desc_act);
}

void ParseFP8QuantConfig(const nlohmann::json &config_json, std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config,
                         QuantConfig &quant_config) {
  quant_config.method = QUANT_FP8_E4M3;
  quant_config.is_checkpoint_fp8_serialized = true;
  quant_config.is_activation_scheme_static = (config_json.at("activation_scheme") == "static");
  if (config_json.contains("weight_block_size") && config_json["weight_block_size"].is_array()) {
    quant_config.is_fp8_blockwise = true;
    quant_config.method = QUANT_BLOCK_FP8_E4M3;
    quant_config.weight_block_size = config_json["weight_block_size"].get<std::vector<size_t>>();
  }
  if (new_deepseek_v3_config->is_moe && quant_config.is_fp8_blockwise == false &&
      !quant_config.is_activation_scheme_static) {
    KLLM_THROW(fmt::format("Not support dynamic fp8 quant_method for moe model."));
  }
  KLLM_LOG_INFO << fmt::format(
      "using quant model, quant method fp8, method type: {}, is_checkpoint_fp8_serialized: {}, "
      "is_activation_scheme_static: {}",
      quant_config.method, quant_config.is_checkpoint_fp8_serialized, quant_config.is_activation_scheme_static);
}

Status NewDeepSeekV3ConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                                   std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config,
                                                   const std::string &yaml_weight_quant_method,
                                                   const std::string &yaml_gptq_backend) {
  new_deepseek_v3_config->is_quant = config_json.contains("quantization_config");
  if (new_deepseek_v3_config->is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "gptq") {
      ParseGPTQQuantConfig(config_json["quantization_config"], new_deepseek_v3_config,
                           new_deepseek_v3_config->quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(config_json["quantization_config"], new_deepseek_v3_config,
                          new_deepseek_v3_config->quant_config);
    } else if (quant_method == "mixed") {
      auto configs = config_json["quantization_config"]["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it) {
        QuantConfig quant_config;
        quant_method = config_json["quantization_config"]["configs"][it.key()]["method"];
        if (quant_method == "gptq") {
          ParseGPTQQuantConfig(config_json["quantization_config"]["configs"][it.key()], new_deepseek_v3_config,
                               quant_config);
        } else if (quant_method == "fp8") {
          ParseFP8QuantConfig(config_json["quantization_config"]["configs"][it.key()], new_deepseek_v3_config,
                              quant_config);
        } else {
          KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
        }
        auto layer_mapping = config_json["quantization_config"]["layer_mapping"][it.key()];
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
  if (new_deepseek_v3_config->is_quant && (new_deepseek_v3_config->quant_config.method == QUANT_GPTQ ||
                                           new_deepseek_v3_config->quant_config.enable_moe_int4)) {
    new_deepseek_v3_config->quant_config.backend = MACHETE_BACKEND;
  } else {
    KLLM_LOG_INFO << "Not using any Quant Backend";
  }

  return Status();
}
}  // namespace ksana_llm
