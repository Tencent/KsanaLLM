/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/environment.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fmt/core.h"
#include "gflags/gflags.h"

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

DEFINE_string(config_file, "examples/ksana_llm.yaml", "The config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");

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
#ifdef ENABLE_BFLOAT16
    return DataType::TYPE_BF16;
#else
    return DataType::TYPE_FP16;
#endif
  } else {
    KLLM_THROW(fmt::format("Not supported model data type: {}.", unified_data_type_raw_str));
  }
}

void ParseGPTQQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config, QuantConfig &quant_config) {
  quant_config.method = QUANT_GPTQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  quant_config.desc_act = config_json.at("desc_act");
  KLLM_LOG_INFO << fmt::format("using quant model, quant method gptq, bits: {}, group_size: {}, desc_act: {}",
                               quant_config.bits, quant_config.group_size, quant_config.desc_act);
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

void Environment::ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                                        std::string &yaml_weight_quant_method, std::string &yaml_gptq_backend) {
  model_config.is_quant = config_json.contains("quantization_config");
  if (model_config.is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "gptq") {
      ParseGPTQQuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "awq") {
      ParseAWQQuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(config_json["quantization_config"], model_config, model_config.quant_config);
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
    model_config.quant_config.backend = MARLIN_BACKEND;
    KLLM_LOG_INFO << "Using MARLIN Quant Backend, only support MARLIN backend in desc_act mode";
  } else if (model_config.quant_config.method == QUANT_GPTQ || model_config.quant_config.method == QUANT_AWQ) {
    if (yaml_gptq_backend == "cutlass") {
      model_config.quant_config.backend = CUTLASS_BACKEND;
      KLLM_LOG_INFO << "Using CUTLASS Quant Backend";
    } else if (yaml_gptq_backend == "marlin") {
      model_config.quant_config.backend = MARLIN_BACKEND;
      KLLM_LOG_INFO << "Using MARLIN Quant Backend";
    } else {
      KLLM_THROW(fmt::format("Not support quant backend {}.", yaml_gptq_backend));
    }
    if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
      // TODO(winminkong): MACHETE_BACKEND will be compatible with all models, int4 matmul layer will be able to
      // automatically select the optimal backend based on conditions such as sm and performance.
      model_config.quant_config.backend = MACHETE_BACKEND;
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
    KLLM_LOG_DEBUG << fmt::format("rope_scaling type: {} factor: {}", model_config.rope_scaling_factor_config.type,
                                  model_config.rope_scaling_factor_config.factor);

    std::unordered_set<std::string> possible_rope_types = {"su", "longrope", "llama3"};
    if (possible_rope_types.find(model_config.rope_scaling_factor_config.type) == possible_rope_types.end()) {
      if (model_config.rope_scaling_factor_config.type == "yarn") {
        derived_max_model_len = rope_scaling_setting.value("original_max_position_embeddings", derived_max_model_len);
        model_config.rope_scaling_factor_config.original_max_position_embeddings =
            rope_scaling_setting.value("original_max_position_embeddings", 32768);
        // for deepseek_yarn config
        if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
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

  model_config.max_token_num = static_cast<int>(derived_max_model_len);
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

Environment::Environment() { dp_group_status_.resize(16, false); }

size_t Environment::GetCommonBlockSize(const ModelConfig &model_config, const PipelineConfig &pipeline_config,
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

std::tuple<size_t, size_t> Environment::GetDeepSeekV3BlockSize(const ModelConfig &model_config,
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
  // The buffer size required for dequantization operations, in bytes.
  size_t convert_size = 0;
  if (block_manager_config.host_allocator_config.kv_cache_dtype == TYPE_FP8_E5M2 ||
      block_manager_config.host_allocator_config.kv_cache_dtype == TYPE_FP8_E4M3) {
    size_t kv_type_size = GetTypeSize(block_manager_config.host_allocator_config.kv_cache_dtype);
    size_t convert_type_size = GetTypeSize(model_configs_[""].weight_data_type);
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

std::tuple<size_t, size_t> Environment::GetCacheBlockSize(const ModelConfig &model_config,
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

Status Environment::ParseConfig(const std::string &config_file, const std::string &model_dir_override) {
  YamlReader yaml_reader;
  const Status status = yaml_reader.LoadFile(config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Load yaml config error." << status.GetMessage();
    return status;
  }

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
  enable_lora_adapter_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_lora_adapter", false);
  embed_tokens_use_cpu_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.embed_tokens_use_cpu", false);
  cuda_graph_ = yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_cuda_graph", false);
  enable_full_shared_expert_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_full_shared_expert", false);
  is_version_report_ = yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.is_version_report", true);
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
    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    pipeline_config.pipeline_para_comm_type = DistributedCommunicationType::SCATTER;
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
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

  // Read profiler config.
  profiler_config_.trace_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.trace_export_url", "");
  profiler_config_.metrics_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.metrics_export_url", "");
  profiler_config_.export_interval_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_interval_millis", 60000);
  profiler_config_.export_timeout_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_timeout_millis", 1000);

  auto attributes = yaml_reader.GetMap(yaml_reader.GetRootNode(), "setting.profiler.attributes");
  for (auto it = attributes.begin(); it != attributes.end(); ++it) {
    const std::string &key = it->first.as<std::string>();
    const std::string &value = it->second.as<std::string>();
    profiler_config_.resource_attributes[key] = value;
  }
  // quantization_config in yaml takes effect when quantization_config in
  // config.json is null.
  yaml_weight_quant_method_ = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.quantization_config.weight.quant_method", "auto");

  yaml_gptq_backend_ = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                          "setting.quantization_config.gptq_backend", "cutlass");

  // Read attn backend config.
  attn_backend_config_.enable_blocked_multi_token_forwarding_kv = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.attn_backend.enable_blocked_multi_token_forwarding_kv", false);
  KLLM_LOG_INFO << "enable_blocked_multi_token_forwarding_kv: "
                << attn_backend_config_.enable_blocked_multi_token_forwarding_kv;

  // Read parallel config.
  expert_parallel_config_.expert_world_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  expert_parallel_config_.expert_para_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);

  // Read base model.
  std::string base_model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  std::string tokenizer_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.tokenizer_dir", "");
  if (tokenizer_dir.empty()) {
    tokenizer_dir = base_model_dir;
  }
  if (!model_dir_override.empty()) {
    base_model_dir = model_dir_override;
    tokenizer_dir = model_dir_override;
  }
  STATUS_CHECK_RETURN(ParseModelConfig(base_model_dir, tokenizer_dir));

  if (model_configs_[""].is_quant == true && model_configs_[""].quant_config.method == QUANT_FP8_E4M3 &&
      model_configs_[""].quant_config.is_checkpoint_fp8_serialized == false) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is fp8_e4m3, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  } else if (model_configs_[""].is_quant == true && model_configs_[""].quant_config.method == QUANT_GPTQ) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is gptq, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  }

  auto kv_cache_dtype_str = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                               "setting.quantization_config.kv_cache.dtype", "auto");
  DataType kv_cache_dtype = model_configs_[""].weight_data_type;

  if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv &&
      !(enable_flash_mla != nullptr && strcmp(enable_flash_mla, "1") == 0 && !IsPrefixCachingEnabled())) {
    if (kv_cache_dtype_str == "fp8_e5m2" || kv_cache_dtype_str == "fp8_e4m3") {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
  } else {
    if (kv_cache_dtype_str == "fp8_e5m2") {
      kv_cache_dtype = TYPE_FP8_E5M2;
    } else if (kv_cache_dtype_str == "fp8_e4m3") {
      kv_cache_dtype = TYPE_FP8_E4M3;
      PrepareKVScales(base_model_dir, model_configs_[""]);
    }
  }

  block_manager_config_.host_allocator_config.kv_cache_dtype = kv_cache_dtype;
  block_manager_config_.device_allocator_config.kv_cache_dtype = kv_cache_dtype;

  InitConnectorConfig(yaml_reader);
  return Status();
}

void Environment::InitConnectorConfig(
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

void Environment::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  block_manager_config_.reserved_device_memory_ratio = reserved_device_memory_ratio;
}

// read GGUF CONFIG
Status Environment::ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config) {
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
    model_config.pad_id = std::any_cast<uint32_t>(get_optional_value("tokenizer.ggml.padding_token_id", (uint32_t)0));
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
    model_config.max_token_num = model_config.max_position_embeddings;

    size_t size_per_head = model_config.hidden_units / model_config.head_num;
    model_config.size_per_head = size_per_head;
    model_config.rotary_embedding = size_per_head;
  } catch (const std::exception &e) {
    return Status(RET_MODEL_INVALID, e.what());
  }

  return Status();
}

Status Environment::ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir,
                                     const std::string &model_config_filename) {
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(model_dir);
  std::filesystem::path abs_tokenizer_dir_path = std::filesystem::absolute(tokenizer_dir);
  std::string config_file = abs_model_dir_path.u8string() + "/" + model_config_filename;
  ModelFileFormat model_file_format;
  ModelConfig model_config;
  Status status;

  model_config.path = abs_model_dir_path.u8string();
  model_config.tokenizer_path = abs_tokenizer_dir_path.u8string();
  model_config.tensor_para_size = tensor_parallel_size_;
  model_config.attn_data_para_size = attn_data_parallel_size_;
  model_config.expert_world_size = expert_world_size_;
  model_config.expert_para_size = expert_parallel_size_;
  model_config.moe_tensor_para_size = tensor_parallel_size_ / expert_parallel_size_;
  model_config.enable_full_shared_expert = enable_full_shared_expert_;

  KLLM_LOG_INFO << fmt::format(
      "ParseModelConfig: tensor_para_size {}, attn_data_para_size {}, expert_world_size {}, expert_para_size {}, "
      "moe_tensor_para_size {}",
      model_config.tensor_para_size, model_config.attn_data_para_size, model_config.expert_world_size,
      model_config.expert_para_size, model_config.moe_tensor_para_size);

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
      PrepareHunyuanLargeAttributes(config_json, model_config);
    } else if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
      PrepareDeepSeekV3Attributes(config_json, model_config);
    } else {
      if (config_json.at("model_type") == "internvl_chat") {
        config_json = config_json.at("llm_config");
      }
      PrepareCommonModelAttributes(config_json, model_config);
    }

    ParseModelMaxLength(config_json, model_config);
    ParseModelQuantConfig(config_json, model_config, yaml_weight_quant_method_, yaml_gptq_backend_);

    UpdateEndIdFromGeneration(model_dir, model_config);
  }

  if (cache_manager_config_.min_flexible_cache_num != 0 && model_config.use_qk_norm) {
    cache_manager_config_.min_flexible_cache_num = 0;
    KLLM_LOG_WARNING << "flexible cache and qk norm cannot be used together, set min_flexible_cache_num to 0";
  }
  if (cuda_graph_ && model_config.is_moe) {
    cuda_graph_ = false;
    KLLM_LOG_WARNING << "moe model cannot be used with cuda graph, set cuda_graph to false";
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
      KLLM_LOG_ERROR << fmt::format(
          "The max_token_num configured in the model's config.json is less than the "
          "max_token_len configured in the ksana yaml file. {} < {}",
          model_config.max_token_num, batch_scheduler_config_.max_token_len);
      return Status(RetCode::RET_INIT_FAILED,
                    fmt::format("Load model config file: {} error. The max_token_num configured in the model's "
                                "config.json is less than the max_token_len configured in the ksana yaml file."
                                " {} < {}",
                                config_file, batch_scheduler_config_.max_token_len, model_config.max_token_num));
    }
    model_config.max_token_num = batch_scheduler_config_.max_token_len;
  }
  batch_scheduler_config_.max_token_len = model_config.max_token_num;
  if ((batch_scheduler_config_.split_fuse_token_num == 0) &&
      (batch_scheduler_config_.max_step_token_num < model_config.max_token_num)) {
    // if no split fuse, request cannot be processed if max_step_num < input token num
    batch_scheduler_config_.max_step_token_num = model_config.max_token_num;
  }
  model_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  model_config.max_batch_size = batch_scheduler_config_.max_batch_size;
  model_config.max_pp_batch_num = batch_scheduler_config_.max_pp_batch_num;
  model_config.max_step_token_num = batch_scheduler_config_.max_step_token_num;
  model_config.k_scales = std::vector<float>(model_config.num_layer + model_config.num_nextn_predict_layers,
                                             1.0f);  // default k scale value
  model_config.v_scales = std::vector<float>(model_config.num_layer + model_config.num_nextn_predict_layers,
                                             1.0f);  // default v scale value

  model_config.enable_prefix_caching = cache_manager_config_.enable_prefix_caching;

  model_configs_[model_config.name] = model_config;

  KLLM_LOG_INFO << fmt::format(
      "Load model {} from config file: {} success. num_layer={}, hidden_units={}, head_num={}, vocab_size={}",
      model_config.name, model_config.path, model_config.num_layer, model_config.hidden_units, model_config.head_num,
      model_config.vocab_size);
  return Status();
}

Status Environment::ParseOptions(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  Status status = ParseConfig(FLAGS_config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("Parse config file {} error: {}", FLAGS_config_file, status.GetMessage());
    return status;
  }

  return Status();
}

Status Environment::InitializeBlockManagerConfig() {
  KLLM_CHECK_WITH_INFO(model_configs_.size() > 0, "No model configed.");
  const ModelConfig &model_config = model_configs_.begin()->second;

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

Status Environment::CheckEnvironment() {
  if (block_manager_config_.host_allocator_config.block_size !=
      block_manager_config_.device_allocator_config.block_size) {
    return Status(RET_INIT_FAILED, fmt::format("block size of device and host is not equal, {} vs {}.",
                                               block_manager_config_.host_allocator_config.block_size,
                                               block_manager_config_.device_allocator_config.block_size));
  }

  return Status();
}

Status Environment::GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs) {
  model_configs = model_configs_;
  return Status();
}

Status Environment::GetModelConfig(const std::string &model_name, ModelConfig &model_config) {
  auto it = model_configs_.find(model_name);
  if (it == model_configs_.end()) {
    return Status(RET_MODEL_NOT_FOUND, fmt::format("No model named {} found.", model_name));
  }

  model_config = it->second;
  return Status();
}

Status Environment::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config = batch_scheduler_config_;
  return Status();
}

Status Environment::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config = cache_manager_config_;
  return Status();
}

Status Environment::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  block_manager_config = block_manager_config_;
  return Status();
}

void Environment::SetBlockManagerConfig(const BlockManagerConfig &block_manager_config) {
  block_manager_config_ = block_manager_config;
}

Status Environment::CalculateBlockNumber() {
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

Status Environment::ResetPipelineBlockNumber() {
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

size_t Environment::GetBlockTokenNum() { return block_manager_config_.device_allocator_config.block_token_num; }

size_t Environment::GetConvertSize() { return block_manager_config_.device_allocator_config.convert_size; }

size_t Environment::GetBlockSize() { return block_manager_config_.device_allocator_config.block_size; }

size_t Environment::GetTotalDeviceBlockNum() { return block_manager_config_.device_allocator_config.blocks_num; }

size_t Environment::GetTotalHostBlockNum() { return block_manager_config_.host_allocator_config.blocks_num; }

DataType Environment::GetKVCacheType() { return block_manager_config_.device_allocator_config.kv_cache_dtype; }

std::vector<int> Environment::GetDataParaGroupDevices(int dp_id) {
  size_t device_count = tensor_parallel_size_;
  size_t group_device_count = device_count / attn_data_parallel_size_;

  std::vector<int> group_devices;
  for (size_t i = 0; i < group_device_count; ++i) {
    group_devices.push_back(dp_id * group_device_count + i);
  }

  return group_devices;
}

size_t Environment::GetAttentionTensorParallel() { return tensor_parallel_size_ / attn_data_parallel_size_; }

void Environment::SetDataParaGroupStatus(int dp_group_id, bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  dp_group_status_[dp_group_id] = enabled;
}

bool Environment::GetDataParaGroupStatus(int dp_group_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return dp_group_status_[dp_group_id];
}

Status Environment::GetEndpointConfig(EndpointConfig &endpoint_config) {
  endpoint_config = endpoint_config_;
  return Status();
}

Status Environment::GetProfilerConfig(ProfilerConfig &profiler_config) {
  profiler_config = profiler_config_;
  return Status();
}

bool Environment::IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

bool Environment::IsFlexibleCachingEnabled() { return cache_manager_config_.min_flexible_cache_num > 0; }

bool Environment::IsSpeculativeDecodingEnabled() { return batch_scheduler_config_.enable_speculative_decoding; }

bool Environment::IsPrefillDecodeSeparation() { return connector_config_.group_role != GroupRole::NONE; }

bool Environment::IsMTPEnabled() { return batch_scheduler_config_.enable_mtp_module; }

size_t Environment::GetTransferLayerChunkSize() { return batch_scheduler_config_.transfer_layer_chunk_size; }

void Environment::InitializeExpertParallelConfig() {
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
