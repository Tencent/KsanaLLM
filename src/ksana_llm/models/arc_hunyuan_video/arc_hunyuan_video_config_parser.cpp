/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/arc_hunyuan_video/arc_hunyuan_video_config_parser.h"

#include "ksana_llm/models/arc_hunyuan_video/arc_hunyuan_video_config.h"
#include "ksana_llm/utils/json_config_utils.h"

namespace ksana_llm {

Status ArcHunyuanVideoConfigParser::ParseModelConfig(const nlohmann::json& config_json,
                                                     const ParallelismBasicConfig& parallel_basic_config,
                                                     const std::string& model_dir,
                                                     std::shared_ptr<BaseModelConfig>& model_config) {
  std::shared_ptr<ArcHunyuanVideoConfig> arc_hunyuan_video_model_config = std::make_shared<ArcHunyuanVideoConfig>();
  model_config = arc_hunyuan_video_model_config;

  arc_hunyuan_video_model_config->weight_data_type = GetModelDataType(config_json);

  // Check if text_config exists
  if (!config_json.contains("text_config")) {
    KLLM_THROW("Config file must contain 'text_config' section for ARC Hunyuan Video model.");
  }

  // Parse text_config section
  const auto& text_config = config_json["text_config"];

  // 以ARC-Hunyuan-Video-7B模型的参数作为默认值
  // Parse core model parameters from text_config
  arc_hunyuan_video_model_config->head_num = text_config.value("num_attention_heads", 32);
  arc_hunyuan_video_model_config->num_key_value_heads = text_config.value("num_key_value_heads", 8);
  arc_hunyuan_video_model_config->inter_size = text_config.value("intermediate_size", 14336);
  arc_hunyuan_video_model_config->vocab_size = text_config.value("vocab_size", 128256);
  arc_hunyuan_video_model_config->num_layer = text_config.value("num_hidden_layers", 32);
  arc_hunyuan_video_model_config->hidden_units = text_config.value("hidden_size", 4096);
  arc_hunyuan_video_model_config->rope_theta = text_config.value("rope_theta", 10000.0f);
  arc_hunyuan_video_model_config->layernorm_eps = text_config.value("rms_norm_eps", 1e-5f);
  arc_hunyuan_video_model_config->start_id = text_config.value("bos_token_id", 127959);
  arc_hunyuan_video_model_config->end_id = text_config.value("eos_token_id", 127960);
  arc_hunyuan_video_model_config->pad_id = text_config.value("pad_token_id", 127961);
  arc_hunyuan_video_model_config->max_position_embeddings = text_config.value("max_position_embeddings", 1024);
  arc_hunyuan_video_model_config->size_per_head = text_config.value("attention_head_dim", 128);
  arc_hunyuan_video_model_config->activation_function = text_config.value("hidden_act", "silu");

  // Parse text_config specific parameters
  arc_hunyuan_video_model_config->attention_bias = text_config.value("attention_bias", false);
  arc_hunyuan_video_model_config->attention_dropout = text_config.value("attention_dropout", 0.0f);
  arc_hunyuan_video_model_config->eod_token_id = text_config.value("eod_token_id", 127957);
  arc_hunyuan_video_model_config->im_end_id = text_config.value("im_end_id", 127963);
  arc_hunyuan_video_model_config->im_newline_id = text_config.value("im_newline_id", 127964);
  arc_hunyuan_video_model_config->im_start_id = text_config.value("im_start_id", 127962);
  arc_hunyuan_video_model_config->image_token_id = text_config.value("image_token_id", 127968);
  arc_hunyuan_video_model_config->initializer_range = text_config.value("initializer_range", 0.02f);
  arc_hunyuan_video_model_config->is_causal = text_config.value("is_causal", true);
  arc_hunyuan_video_model_config->mlp_bias = text_config.value("mlp_bias", false);
  arc_hunyuan_video_model_config->norm_type = text_config.value("norm_type", "hf_rms");
  arc_hunyuan_video_model_config->num_media_embeds = text_config.value("num_media_embeds", 257);
  arc_hunyuan_video_model_config->position_embedding_xdrope = text_config.value("position_embedding_xdrope", true);
  arc_hunyuan_video_model_config->use_qk_norm = text_config.value("use_qk_norm", true);
  arc_hunyuan_video_model_config->use_rotary_pos_emb = text_config.value("use_rotary_pos_emb", true);

  const auto& rope_scaling = text_config["rope_scaling"];
  arc_hunyuan_video_model_config->rope_scaling_factor_config.type = rope_scaling.value("type", "dynamic");
  arc_hunyuan_video_model_config->rope_scaling_factor_config.factor = rope_scaling.value("factor", 1.0f);
  arc_hunyuan_video_model_config->rope_scaling_factor_config.scaling_alpha = rope_scaling.value("alpha", 1000.0f);
  arc_hunyuan_video_model_config->rope_scaling_factor_config.beta_fast = rope_scaling.value("beta_fast", 32.0f);
  arc_hunyuan_video_model_config->rope_scaling_factor_config.beta_slow = rope_scaling.value("beta_slow", 1.0f);
  arc_hunyuan_video_model_config->rope_scaling_factor_config.mscale = rope_scaling.value("mscale", 1.0f);
  arc_hunyuan_video_model_config->rope_scaling_factor_config.mscale_all_dim =
      rope_scaling.value("mscale_all_dim", 1.0f);
  std::vector<float> float_xdrope_section = text_config["xdrope_section"].get<std::vector<float>>();

  // Check for tie_word_embeddings in root config
  if (!config_json.contains("tie_word_embeddings")) {
    arc_hunyuan_video_model_config->exist_tie_embeddings_param = false;
  } else {
    arc_hunyuan_video_model_config->tie_word_embeddings = config_json["tie_word_embeddings"];
  }

  arc_hunyuan_video_model_config->head_groups =
      arc_hunyuan_video_model_config->head_num / arc_hunyuan_video_model_config->num_key_value_heads;

  // xdrope_section从float转换成int，并转换为cumsum方便算子使用
  std::vector<int> int_xdrope_section(float_xdrope_section.size());
  for (size_t i = 0; i < int_xdrope_section.size(); i++) {
    int_xdrope_section[i] =
        static_cast<int>(float_xdrope_section[i] * arc_hunyuan_video_model_config->size_per_head / 2);
  }
  for (size_t i = 1; i < int_xdrope_section.size(); i++) {
    int_xdrope_section[i] += int_xdrope_section[i - 1];
  }
  arc_hunyuan_video_model_config->rope_scaling_factor_config.xdrope_section = int_xdrope_section;

  return Status();
}

}  // namespace ksana_llm
