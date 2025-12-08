/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_model_config_parser.h"

#include <memory>

#include "ksana_llm/models/llama/llama_model_config.h"
#include "ksana_llm/utils/gguf_file_utils.h"
#include "ksana_llm/utils/json_config_utils.h"

namespace ksana_llm {

LlamaModelConfigParser::LlamaModelConfigParser() {}

LlamaModelConfigParser::~LlamaModelConfigParser() {}

Status LlamaModelConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                const ParallelismBasicConfig &parallel_basic_config,
                                                const std::string &model_dir,
                                                std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<LlamaModelConfig> llama_model_config = std::make_shared<LlamaModelConfig>();
  model_config = llama_model_config;

  llama_model_config->weight_data_type = GetModelDataType(config_json);

  // Use llama-7B config as default values
  llama_model_config->head_num = config_json.value("num_attention_heads", 32);

  llama_model_config->num_key_value_heads = config_json.value("num_key_value_heads", llama_model_config->head_num);

  llama_model_config->inter_size = config_json.value("intermediate_size", 11008);

  llama_model_config->vocab_size = config_json.value("vocab_size", 32000);

  llama_model_config->num_layer = config_json.value("num_hidden_layers", 32);

  llama_model_config->hidden_units = config_json.value("hidden_size", 4096);

  llama_model_config->rope_theta = config_json.value("rope_theta", 10000.0f);

  llama_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);

  llama_model_config->layernorm_eps = config_json.value("layer_norm_epsilon", llama_model_config->layernorm_eps);

  llama_model_config->start_id = config_json.value("bos_token_id", 1);

  // for llama3.1 config
  if (config_json.contains("eos_token_id") && config_json["eos_token_id"].is_array()) {
    llama_model_config->end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    llama_model_config->end_ids =
        std::vector<uint32_t>{static_cast<unsigned int>(config_json.value("eos_token_id", 2))};
  }

  llama_model_config->pad_id = config_json.value("pad_token_id", 0);
  llama_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 2048);
  if (!config_json.contains("tie_word_embeddings")) {
    llama_model_config->exist_tie_embeddings_param = false;
  }
  llama_model_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  llama_model_config->is_visual = config_json.contains("visual");

  size_t size_per_head = llama_model_config->hidden_units / llama_model_config->head_num;
  llama_model_config->size_per_head = size_per_head;
  llama_model_config->rotary_embedding = size_per_head;

  // TODO(huicongyao): support quant config parse, and model weight process
  llama_model_config->is_quant = config_json.contains("quantization_config");
  return Status();
}

DataType GetGGUFWeightDataType(uint32_t gguf_model_file_type) {
  switch (gguf_model_file_type) {
    case NewGGUFModelFileType::NEW_LLAMA_FTYPE_ALL_F32:
      return DataType::TYPE_FP32;
    case NewGGUFModelFileType::NEW_LLAMA_FTYPE_MOSTLY_F16:
      return DataType::TYPE_FP16;
    case NewGGUFModelFileType::NEW_LLAMA_FTYPE_MOSTLY_BF16:
      return DataType::TYPE_BF16;
    default:
      return TYPE_INVALID;
  }
}

Status LlamaModelConfigParser::ParseModelConfig(const std::unordered_map<std::string, NewGGUFMetaValue> &gguf_meta,
                                                std::shared_ptr<BaseModelConfig> &model_config) {
  // Create real model config.
  std::shared_ptr<LlamaModelConfig> llama_model_config = std::make_shared<LlamaModelConfig>();
  model_config = llama_model_config;

  llama_model_config->weight_data_type = GetGGUFWeightDataType(std::any_cast<uint32_t>(
      GetValueFromGGUFMeta(gguf_meta, "general.file_type", NewGGUFModelFileType::NEW_LLAMA_FTYPE_MOSTLY_F16)));

  std::string model_type = std::any_cast<std::string>(GetValueFromGGUFMeta(gguf_meta, "general.architecture"));

  llama_model_config->head_num =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".attention.head_count"));

  llama_model_config->num_key_value_heads =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".attention.head_count_kv"));

  llama_model_config->inter_size =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".feed_forward_length"));

  llama_model_config->vocab_size = std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".vocab_size"));

  llama_model_config->num_layer = std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".block_count"));

  llama_model_config->hidden_units =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".embedding_length"));

  llama_model_config->rope_theta =
      std::any_cast<float>(GetValueFromGGUFMeta(gguf_meta, model_type + ".rope.freq_base", 10000.0f));

  llama_model_config->layernorm_eps =
      std::any_cast<float>(GetValueFromGGUFMeta(gguf_meta, model_type + ".attention.layer_norm_rms_epsilon", 1e-6));

  llama_model_config->start_id =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, "tokenizer.ggml.bos_token_id", 1));

  llama_model_config->pad_id = std::any_cast<uint32_t>(
      GetValueFromGGUFMeta(gguf_meta, "tokenizer.ggml.padding_token_id", static_cast<uint32_t>(0)));

  llama_model_config->max_position_embeddings =
      std::any_cast<uint32_t>(GetValueFromGGUFMeta(gguf_meta, model_type + ".context_length", 2048));

  llama_model_config->tie_word_embeddings =
      std::any_cast<bool>(GetValueFromGGUFMeta(gguf_meta, model_type + ".tie_word_embeddings", false));

  llama_model_config->is_visual = gguf_meta.count("visual");

  // Handle 'end_ids' which might be a single value or an array
  if (gguf_meta.count("tokenizer.ggml.eos_token_id")) {
    auto eos_token_meta = gguf_meta.at("tokenizer.ggml.eos_token_id");
    if (eos_token_meta.type == NewGGUFMetaValueType::NEW_GGUF_METADATA_VALUE_TYPE_ARRAY) {
      llama_model_config->end_ids = std::any_cast<std::vector<uint32_t>>(eos_token_meta.value);
    } else {
      llama_model_config->end_ids = {std::any_cast<uint32_t>(eos_token_meta.value)};
    }
  } else {
    llama_model_config->end_ids = {2};
  }
  llama_model_config->max_token_num = llama_model_config->max_position_embeddings;

  size_t size_per_head = llama_model_config->hidden_units / llama_model_config->head_num;
  llama_model_config->size_per_head = size_per_head;
  llama_model_config->rotary_embedding = size_per_head;

  return Status();
}

}  // namespace ksana_llm
