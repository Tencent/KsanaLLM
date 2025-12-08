/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/config/model_config_parser.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "fmt/core.h"

#include "ksana_llm/models/bge_reranker_minicpm/bge_reranker_minicpm_config.h"
#include "ksana_llm/models/chatglm/chatglm_config.h"
#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_config.h"
#include "ksana_llm/models/gpt/gpt_config.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

DataType GetModelDataType(const nlohmann::json &config_json, ModelConfig &model_config) {
  std::string data_type_raw_str = config_json.value("torch_dtype", "float16");
  std::string unified_data_type_raw_str = data_type_raw_str;
  // unify it to lower case
  std::transform(unified_data_type_raw_str.begin(), unified_data_type_raw_str.end(), unified_data_type_raw_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (unified_data_type_raw_str == "float16") {
    return DataType::TYPE_FP16;
  } else if (unified_data_type_raw_str == "bfloat16") {
    return DataType::TYPE_BF16;
  } else {
    KLLM_THROW(fmt::format("Not supported model data type: {}.", unified_data_type_raw_str));
  }
}

void ParseGPTQQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config, QuantConfig &quant_config) {
  quant_config.method = QUANT_GPTQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  quant_config.desc_act = config_json.value("desc_act", false);
  quant_config.input_scale = config_json.value("input_scale", false);
  KLLM_LOG_INFO << fmt::format(
      "using quant model, quant method gptq, bits: {}, group_size: {}, desc_act: {}, input_scale:{}", quant_config.bits,
      quant_config.group_size, quant_config.desc_act, quant_config.input_scale);
}

void ParseAWQQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config, QuantConfig &quant_config) {
  if (model_config.is_moe) {
    KLLM_THROW(fmt::format("Not support quant_method awq for moe model."));
  }
  quant_config.method = QUANT_AWQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  KLLM_LOG_INFO << fmt::format("using quant model, quant method awq, bits: {}, group_size: {}", quant_config.bits,
                               quant_config.group_size);
}

void ParseFP8QuantConfig(const nlohmann::json &config_json, ModelConfig &model_config, QuantConfig &quant_config) {
  quant_config.method = QUANT_FP8_E4M3;
  quant_config.is_checkpoint_fp8_serialized = true;
  quant_config.is_activation_scheme_static = (config_json.at("activation_scheme") == "static");
  if (config_json.contains("weight_block_size") && config_json["weight_block_size"].is_array()) {
    quant_config.is_fp8_blockwise = true;
    quant_config.method = QUANT_BLOCK_FP8_E4M3;
    quant_config.weight_block_size = config_json["weight_block_size"].get<std::vector<size_t>>();
  }
  if (model_config.is_moe && quant_config.is_fp8_blockwise == false && !quant_config.is_activation_scheme_static) {
    KLLM_THROW(fmt::format("Not support dyanmic fp8 quant_method for moe model."));
  }
  KLLM_LOG_INFO << fmt::format(
      "using quant model, quant method fp8, method type: {}, is_checkpoint_fp8_serialized: {}, "
      "is_activation_scheme_static: {}",
      quant_config.method, quant_config.is_checkpoint_fp8_serialized, quant_config.is_activation_scheme_static);
}

void ParseModelOptQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config, QuantConfig &quant_config) {
  std::string algo = config_json.value("quant_algo", "");
  if (algo == "W4A8_AWQ") {
    quant_config.method = QUANT_W4A8_AWQ;
    KLLM_LOG_INFO << "using quant model, quant method W4A8_AWQ";
  } else {
    KLLM_THROW("only support W4A8_AWQ algo in modelopt");
  }
}

void EnvModelConfigParser::ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                                                 std::string &yaml_weight_quant_method,
                                                 std::string &yaml_gptq_backend) {
  model_config.is_quant = config_json.contains("quantization_config");
  if (model_config.is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "gptq") {
      ParseGPTQQuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "awq") {
      ParseAWQQuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "modelopt") {
      ParseModelOptQuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "mixed") {
      auto configs = config_json["quantization_config"]["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it) {
        QuantConfig quant_config;
        quant_method = config_json["quantization_config"]["configs"][it.key()]["method"];
        if (quant_method == "gptq") {
          ParseGPTQQuantConfig(config_json["quantization_config"]["configs"][it.key()], model_config, quant_config);
        } else if (quant_method == "awq") {
          ParseAWQQuantConfig(config_json["quantization_config"]["configs"][it.key()], model_config, quant_config);
        } else if (quant_method == "fp8") {
          ParseFP8QuantConfig(config_json["quantization_config"]["configs"][it.key()], model_config, quant_config);
        } else {
          KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
        }
        auto layer_mapping = config_json["quantization_config"]["layer_mapping"][it.key()];
        quant_config.pattern_layers = layer_mapping["pattern_layers"].get<std::vector<std::string>>();
        quant_config.ignored_layers = layer_mapping["ignored_layers"].get<std::vector<std::string>>();
        if (layer_mapping["default_config"]) {
          model_config.quant_config = quant_config;
        } else {
          model_config.sub_quant_configs.push_back(quant_config);
        }
      }
      if (model_config.sub_quant_configs.size() == 1 && model_config.sub_quant_configs[0].method == QUANT_GPTQ &&
          model_config.sub_quant_configs[0].pattern_layers.size() == 1 &&
          model_config.sub_quant_configs[0].pattern_layers[0] == ".mlp.experts.") {
        model_config.quant_config.enable_moe_int4 = true;
      }
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto") {
    if (model_config.is_moe) {
      KLLM_THROW(fmt::format("Not support quant_method {} for moe model.", yaml_weight_quant_method));
    }
    if (yaml_weight_quant_method == "fp8_e4m3") {
      // when quantization_config in config.json is null,
      // quant method is decided by quantization_config in yaml.
      model_config.is_quant = true;
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = false;
      model_config.quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, model_config.quant_config.is_checkpoint_fp8_serialized,
          model_config.quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
    }
  }

  if (model_config.quant_config.method == QUANT_GPTQ && model_config.quant_config.desc_act == true) {
    model_config.quant_config.backend = MARLIN_LINEAR_BACKEND;
    KLLM_LOG_INFO << "Using MARLIN Quant Backend, only support MARLIN backend in desc_act mode";
  } else if (model_config.quant_config.method == QUANT_GPTQ || model_config.quant_config.method == QUANT_AWQ) {
    if (yaml_gptq_backend == "cutlass") {
      model_config.quant_config.backend = CUTLASS_LINEAR_BACKEND;
      KLLM_LOG_INFO << "Using CUTLASS Quant Backend";
    } else if (yaml_gptq_backend == "marlin") {
      model_config.quant_config.backend = MARLIN_LINEAR_BACKEND;
      KLLM_LOG_INFO << "Using MARLIN Quant Backend";
    } else {
      KLLM_THROW(fmt::format("Not support quant backend {}.", yaml_gptq_backend));
    }
    if (model_config.use_mla) {
      // TODO(winminkong): MACHETE_LINEAR_BACKEND will be compatible with all models, int4 matmul layer will be able to
      // automatically select the optimal backend based on conditions such as sm and performance.
      model_config.quant_config.backend = MACHETE_LINEAR_BACKEND;
      KLLM_LOG_INFO << "Using MACHETE Quant Backend, DeepSeek only support MACHETE backend at present";
    }
  } else {
    KLLM_LOG_INFO << "Not using any Quant Backend";
  }

  if (model_config.type == "hunyuan" && config_json.contains("use_mixed_mlp_moe") && config_json["use_mixed_mlp_moe"]) {
    if (model_config.quant_config.method == QUANT_GPTQ && model_config.weight_data_type != TYPE_FP16) {
      KLLM_THROW("Only support QUANT_GPTQ with data_type fp16 for HunyuanLarge.");
    }
    if (model_config.quant_config.method == QUANT_AWQ) {
      KLLM_THROW("Not support QUANT_AWQ for HunyuanLarge.");
    }
  }
}

void ParseModelMaxLength(const nlohmann::json &config_json, ModelConfig &model_config) {
  // refer to
  // github vllm-project/vllm/blob vllm/config.py#L1116
  float derived_max_model_len = std::numeric_limits<float>::infinity();
  std::vector<std::string> possible_keys = {/* OPT */ "max_position_embeddings",
                                            /* GPT-2 */ "n_positions",
                                            /* MPT */ "max_seq_len",
                                            /* ChatGLM2 */ "seq_length",
                                            /* Command-R */ "model_max_length",
                                            /* Others */ "max_sequence_length",
                                            "max_seq_length",
                                            "seq_len"};
  for (std::string &key : possible_keys) {
    float max_len = config_json.value(key, std::numeric_limits<float>::infinity());
    derived_max_model_len = std::min(derived_max_model_len, max_len);
  }
  if (derived_max_model_len == std::numeric_limits<float>::infinity()) {
    std::string possible_keys_str = Vector2Str<std::string>(possible_keys);
    KLLM_THROW(
        fmt::format("The model's config.json does not contain any of the following keys to determine"
                    " the original maximum length of the model: {}",
                    possible_keys_str));
  }

  auto rope_scaling_setting = config_json.value("rope_scaling", nlohmann::json());
  if (!rope_scaling_setting.is_null()) {
    model_config.rope_scaling_factor_config.type = rope_scaling_setting.value("type", "default");
    // fit llama3.1 config
    model_config.rope_scaling_factor_config.type =
        rope_scaling_setting.value("rope_type", model_config.rope_scaling_factor_config.type);
    model_config.rope_scaling_factor_config.factor = rope_scaling_setting.value("factor", 1.0f);
    // adjust the rope_scaling type based on the mrope_section
    if (rope_scaling_setting.contains("mrope_section") && model_config.rope_scaling_factor_config.type != "mrope") {
      KLLM_LOG_DEBUG << fmt::format("Replace rope_scaling type {} with mrope",
                                    model_config.rope_scaling_factor_config.type);
      model_config.rope_scaling_factor_config.type = "mrope";
    }
    KLLM_LOG_DEBUG << fmt::format("rope_scaling type: {} factor: {}", model_config.rope_scaling_factor_config.type,
                                  model_config.rope_scaling_factor_config.factor);

    std::unordered_set<std::string> possible_rope_types = {"su", "longrope", "llama3"};
    if (possible_rope_types.find(model_config.rope_scaling_factor_config.type) == possible_rope_types.end()) {
      if (model_config.rope_scaling_factor_config.type == "yarn") {
        derived_max_model_len = rope_scaling_setting.value("original_max_position_embeddings", derived_max_model_len);
        model_config.rope_scaling_factor_config.original_max_position_embeddings =
            rope_scaling_setting.value("original_max_position_embeddings", 32768);
        // for deepseek_yarn config
        if (model_config.use_mla) {
          // deepseek v2 and v3 have the same yarn implementation
          model_config.rope_scaling_factor_config.use_deepseek_yarn = true;
        }
        model_config.rope_scaling_factor_config.beta_fast = rope_scaling_setting.value("beta_fast", 32.0f);
        model_config.rope_scaling_factor_config.beta_slow = rope_scaling_setting.value("beta_slow", 1.0f);
        model_config.rope_scaling_factor_config.mscale = rope_scaling_setting.value("mscale", 1.0f);
        model_config.rope_scaling_factor_config.mscale_all_dim = rope_scaling_setting.value("mscale_all_dim", 1.0f);
      }
      // for dynamic alpha
      if (model_config.rope_scaling_factor_config.type == "dynamic" && rope_scaling_setting.contains("alpha")) {
        model_config.rope_scaling_factor_config.has_alpha = true;
        model_config.rope_scaling_factor_config.scaling_alpha = rope_scaling_setting.value("alpha", 1.0f);
      } else {
        derived_max_model_len *= model_config.rope_scaling_factor_config.factor;
      }
    }

    if (model_config.rope_scaling_factor_config.type == "llama3") {
      model_config.rope_scaling_factor_config.low_freq_factor = rope_scaling_setting.value("low_freq_factor", 1.0f);
      model_config.rope_scaling_factor_config.high_freq_factor = rope_scaling_setting.value("high_freq_factor", 4.0f);
      model_config.rope_scaling_factor_config.original_max_position_embeddings =
          rope_scaling_setting.value("original_max_position_embeddings", 8192);
    }

    if (model_config.rope_scaling_factor_config.type == "mrope") {
      auto &mrope_section = model_config.rope_scaling_factor_config.mrope_section;
      mrope_section = rope_scaling_setting["mrope_section"].get<std::vector<int>>();
      KLLM_CHECK_WITH_INFO(mrope_section.size() == 3,
                           "The length of mrope section used for multimodal rotary embedding must be 3.");
      // Perform a prefix sum to facilitate the MRotaryEmbedding kernel.
      for (int i = 1; i < 3; i++) {
        mrope_section[i] += mrope_section[i - 1];
      }
    }

    // InternLM2 use InternLM2RotaryEmbedding
    // It modifies the initialization method of "base" based on the "dynamic" approach.
    if (model_config.type == "internlm2" || model_config.type == "internlmxcomposer2" ||
        model_config.type == "internvl_chat") {
      KLLM_LOG_DEBUG << "InternLM2 Model use InternLM2RotaryEmbedding";
      model_config.rope_scaling_factor_config.type = "internlm2_dynamic";
    }
  }

  model_config.max_training_seq_len = static_cast<int>(derived_max_model_len);
}

void UpdateEndIdFromGeneration(const std::string &model_dir, ModelConfig &model_config) {
  // Priority: `generation_config` argument > `config.json` argument
  // It is recommended to set all generation parameters in `generation_config`
  // Refer to
  // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1736
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(model_dir);
  std::string config_file = abs_model_dir_path.u8string() + "/generation_config.json";

  nlohmann::json config_json;
  std::ifstream file(config_file);
  if (!file.is_open()) {
    KLLM_LOG_DEBUG << fmt::format("Gneration config file: {} does not exist.", config_file);
    return;
  } else {
    file >> config_json;
    file.close();
  }

  if (!config_json.contains("eos_token_id")) {
    return;
  }

  std::vector<uint32_t> end_ids;
  if (config_json.at("eos_token_id").is_array()) {
    end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    end_ids = std::vector<uint32_t>{config_json.at("eos_token_id")};
  }
  if (end_ids != model_config.end_ids) {
    KLLM_LOG_WARNING << fmt::format("eos_token_id: [{}] in model config is overwritten by [{}] in generation config",
                                    fmt::join(model_config.end_ids, ", "), fmt::join(end_ids, ", "));
    model_config.end_ids = std::move(end_ids);
  }
}

// read GGUF CONFIG
Status EnvModelConfigParser::ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config) {
  // load meta data from GGUF file
  GGUFFileTensorLoader gguf_loader(meta_file_path, model_config.load_bias);
  auto context = gguf_loader.GetMetadata();
  auto &metadata_map = context->metadata_map;

  // Helper functions to retrieve metadata values
  auto get_required_value = [&](const std::string &key, const std::string &error_msg) -> std::any {
    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) {
      return it->second.value;
    } else {
      throw std::runtime_error(error_msg);
    }
  };

  auto get_optional_value = [&](const std::string &key, const std::any &default_value) -> std::any {
    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) {
      return it->second.value;
    } else {
      return default_value;
    }
  };

  try {
    model_config.type = std::any_cast<std::string>(
        get_required_value("general.architecture", "Model type is not supported in GGUF format."));
    if (model_config.type != "llama") {
      throw std::runtime_error("Model type is not supported in GGUF format.");
    }

    std::string model_type = model_config.type;
    uint32_t ftype =
        std::any_cast<uint32_t>(get_optional_value("general.file_type", GGUFModelFileType::LLAMA_FTYPE_MOSTLY_F16));
    model_config.weight_data_type = GGUFFileTensorLoader::ConverGGUFModelFileTypeToDataType(ftype);
    model_config.head_num = std::any_cast<uint32_t>(
        get_required_value(model_type + ".attention.head_count", "Model head_num is not supported in GGUF format."));
    model_config.num_key_value_heads = std::any_cast<uint32_t>(get_required_value(
        model_type + ".attention.head_count_kv", "Model num_key_value_heads is not supported in GGUF format."));
    model_config.inter_size = std::any_cast<uint32_t>(
        get_required_value(model_type + ".feed_forward_length", "Model inter_size is not supported in GGUF format."));
    model_config.vocab_size = std::any_cast<uint32_t>(
        get_required_value(model_type + ".vocab_size", "Model vocab_size is not supported in GGUF format."));
    model_config.num_layer = std::any_cast<uint32_t>(
        get_required_value(model_type + ".block_count", "Model num_layer is not supported in GGUF format."));
    model_config.hidden_units = std::any_cast<uint32_t>(
        get_required_value(model_type + ".embedding_length", "Model hidden_units is not supported in GGUF format."));
    model_config.rope_theta = std::any_cast<float>(get_optional_value(model_type + ".rope.freq_base", 10000.0f));
    model_config.layernorm_eps =
        std::any_cast<float>(get_optional_value(model_type + ".attention.layer_norm_rms_epsilon", 1e-6));
    model_config.start_id = std::any_cast<uint32_t>(get_optional_value("tokenizer.ggml.bos_token_id", 1));
    model_config.pad_id =
        std::any_cast<uint32_t>(get_optional_value("tokenizer.ggml.padding_token_id", static_cast<uint32_t>(0)));
    model_config.max_position_embeddings =
        std::any_cast<uint32_t>(get_optional_value(model_type + ".context_length", 2048));
    model_config.tie_word_embeddings =
        std::any_cast<bool>(get_optional_value(model_type + ".tie_word_embeddings", false));
    model_config.is_visual = metadata_map.count("visual");

    // Handle 'end_ids' which might be a single value or an array
    if (metadata_map.count("tokenizer.ggml.eos_token_id")) {
      auto eos_token_meta = metadata_map["tokenizer.ggml.eos_token_id"];
      if (eos_token_meta.type == GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_ARRAY) {
        model_config.end_ids = std::any_cast<std::vector<uint32_t>>(eos_token_meta.value);
      } else {
        model_config.end_ids = {std::any_cast<uint32_t>(eos_token_meta.value)};
      }
    } else {
      model_config.end_ids = {2};
    }
    model_config.max_training_seq_len = model_config.max_position_embeddings;

    size_t size_per_head = model_config.hidden_units / model_config.head_num;
    model_config.size_per_head = size_per_head;
    model_config.rotary_embedding = size_per_head;
  } catch (const std::exception &e) {
    return Status(RET_MODEL_INVALID, e.what());
  }

  return Status();
}

Status EnvModelConfigParser::ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir,
                                              const std::string &model_config_filename, ModelConfig &model_config) {
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(model_dir);
  std::filesystem::path abs_tokenizer_dir_path = std::filesystem::absolute(tokenizer_dir);
  std::string config_file = abs_model_dir_path.u8string() + "/" + model_config_filename;
  ModelFileFormat model_file_format;
  Status status;

  model_config.path = abs_model_dir_path.u8string();
  model_config.tokenizer_path = abs_tokenizer_dir_path.u8string();

  std::vector<std::string> weights_file_list = SearchLocalPath(model_dir, model_file_format);
  model_config.model_file_format = model_file_format;

  if (model_file_format == GGUF) {
    status = ParseModelConfigFromGGUF(weights_file_list[0], model_config);
    if (!status.OK()) {
      return status;
    }
  } else {
    nlohmann::json config_json;
    std::ifstream file(config_file);
    if (!file.is_open()) {
      KLLM_LOG_ERROR << fmt::format("Load model config file: {} error.", config_file);
      return Status(RetCode::RET_MODEL_INVALID, fmt::format("Load model config file: {} error.", config_file));
    } else {
      file >> config_json;
      file.close();
    }

    model_config.weight_data_type = GetModelDataType(config_json, model_config);
    model_config.type = config_json.at("model_type");
    auto architectures = config_json.at("architectures");

    if (model_config.type == "internlm2") {
      if (std::find(architectures.begin(), architectures.end(), "InternLMXComposer2ForCausalLM") !=
          architectures.end()) {
        model_config.type = "internlmxcomposer2";
        KLLM_LOG_INFO << "model type changed from internlm2 to internlmxcomposer2";
      }
    }

    if (model_config.type == "internvl_chat") {
      if (std::find(architectures.begin(), architectures.end(), "InternVLChatModel") != architectures.end()) {
        auto llm_architectures = config_json.at("llm_config").at("architectures");
        if (std::find(llm_architectures.begin(), llm_architectures.end(), "Qwen2ForCausalLM") !=
            llm_architectures.end()) {
          // internvl_qwen2 shares the same model architecture as qwen2
          // but different weight name from model.safetensors
          model_config.type = "internvl_qwen2";
          KLLM_LOG_INFO << "model type changed from internlm2 to internvl_qwen2";
        }
      }
    }
    if (model_config.type == "qwen2_5_vl") {
      model_config.type = "qwen2_vl";
      KLLM_LOG_INFO << "model type changed from qwen2_5_vl to qwen2_vl";
    }

    if (model_config.type == "chatglm") {
      PrepareChatglmAttributes(config_json, model_config);
    } else if (model_config.type == "openai-gpt") {  // GPT-1
      // For fairseq transformer, we use the same config as huggingface openai-gpt, and distinguish them by the vocab
      // size.
      if (config_json.at("vocab_size") == 7000) {
        model_config.type = "fairseq-transformer";
        PrepareFairseqTransformerAttributes(config_json, model_config);
      } else {
        PrepareGPT1Attributes(config_json, model_config);
      }
    } else if (model_config.type == "gpt2") {
      PrepareGPT2Attributes(config_json, model_config);
    } else if (model_config.type == "qwen2_moe" || model_config.type == "qwen3_moe") {
      PrepareQwenMoeAttributes(config_json, model_config);
    } else if (model_config.type == "llama4") {
      config_json = config_json["text_config"];
      model_config.weight_data_type = GetModelDataType(config_json, model_config);
      PrepareLlama4Attributes(config_json, model_config);
    } else if (model_config.type == "mixtral") {
      PrepareMixtralAttributes(config_json, model_config);
    } else if (model_config.type == "hunyuan") {
      if (config_json.contains("use_mixed_mlp_moe") && config_json["use_mixed_mlp_moe"]) {
        PrepareHunyuanLargeAttributes(config_json, model_config);
      } else {
        PrepareHunyuanTurboAttributes(config_json, model_config);
      }
    } else if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v32" ||
               model_config.type == "deepseek_v2" || model_config.type == "kimi_k2") {
      PrepareDeepSeekV3Attributes(config_json, model_config);
    } else if (model_config.type == "minicpm") {
      PrepareBgeRerankerMinicpmAttributes(config_json, model_config);
    } else {
      if (config_json.at("model_type") == "internvl_chat") {
        config_json = config_json.at("llm_config");
      }
      PrepareCommonModelAttributes(config_json, model_config);
    }

    ParseModelMaxLength(config_json, model_config);
    ParseModelQuantConfig(config_json, model_config, weight_quant_method_, gptq_backend_);

    UpdateEndIdFromGeneration(model_dir, model_config);
  }

  model_config.k_scales = std::vector<float>(model_config.num_layer + model_config.num_nextn_predict_layers,
                                             1.0f);  // default k scale value
  model_config.v_scales = std::vector<float>(model_config.num_layer + model_config.num_nextn_predict_layers,
                                             1.0f);  // default v scale value

  return Status();
}
}  // namespace ksana_llm
