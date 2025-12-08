/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen/new_qwen_config_parser.h"

#include <memory>

#include "ksana_llm/models/qwen/new_qwen_config.h"
#include "ksana_llm/utils/json_config_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

NewQwenConfigParser::NewQwenConfigParser() {}

NewQwenConfigParser::~NewQwenConfigParser() {}

Status NewQwenConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                             const ParallelismBasicConfig &parallel_basic_config,
                                             std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<NewQwenConfig> new_qwen_model_config = std::make_shared<NewQwenConfig>();
  model_config = new_qwen_model_config;

  new_qwen_model_config->weight_data_type = GetModelDataType(config_json);

  // Use Qwen3-32B config as default values
  new_qwen_model_config->head_num = config_json.value("num_attention_heads", 64);
  new_qwen_model_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  new_qwen_model_config->inter_size = config_json.value("intermediate_size", 25600);
  new_qwen_model_config->vocab_size = config_json.value("vocab_size", 151936);
  new_qwen_model_config->num_layer = config_json.value("num_hidden_layers", 64);
  new_qwen_model_config->hidden_units = config_json.value("hidden_size", 5120);
  new_qwen_model_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  new_qwen_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  new_qwen_model_config->start_id = config_json.value("bos_token_id", 151643);
  new_qwen_model_config->end_id = config_json.value("eos_token_id", 151645);
  new_qwen_model_config->pad_id = config_json.value("pad_token_id", 0);
  new_qwen_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 40960);
  if (!config_json.contains("tie_word_embeddings")) {
    new_qwen_model_config->exist_tie_embeddings_param = false;
  }
  new_qwen_model_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  new_qwen_model_config->is_visual = config_json.contains("visual");

  if (config_json.contains("head_dim")) {
    new_qwen_model_config->size_per_head = config_json.value("head_dim", 128);
  } else {  // for config.json which doesn't contain head_dim, init size_per_head during weight processing
    new_qwen_model_config->size_per_head = 0;
  }

  // parse quantization config
  auto env = Singleton<Environment>::GetInstance();
  ParseQuantConfig(config_json, new_qwen_model_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

void ParseModelOptQuantConfig(const nlohmann::json &config_json, std::shared_ptr<NewQwenConfig> new_qwen_config,
                              QuantConfig &quant_config) {
  std::string algo = config_json.value("quant_algo", "");
  if (algo == "W4A8_AWQ") {
    quant_config.method = QUANT_W4A8_AWQ;
    KLLM_LOG_INFO << "using quant model, quant method W4A8_AWQ";
  } else {
    KLLM_THROW("only support W4A8_AWQ algo in modelopt");
  }
}

Status NewQwenConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                             std::shared_ptr<NewQwenConfig> new_qwen_config,
                                             const std::string &yaml_weight_quant_method,
                                             const std::string &yaml_gptq_backend) {
  new_qwen_config->is_quant = config_json.contains("quantization_config");
  if (new_qwen_config->is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "modelopt") {
      ParseModelOptQuantConfig(config_json["quantization_config"], new_qwen_config, new_qwen_config->quant_config);
    } else {
      KLLM_THROW(fmt::format("Not support quant method: {}", quant_method));
    }
  }
  return Status();
}

}  // namespace ksana_llm
