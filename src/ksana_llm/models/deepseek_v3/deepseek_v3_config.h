/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareDeepSeekV3Attributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  // For moe config
  model_config.moe_config.num_experts = config_json.value("n_routed_experts", 256);
  if (model_config.moe_config.num_experts > 1) {
    if (model_config.type == "deepseek_v3" || model_config.type == "deepseek_v2") {
      model_config.moe_config.use_vllm_moe = true;
    }
    model_config.is_moe = true;
    model_config.moe_config.moe_inter_size = config_json.value("moe_intermediate_size", model_config.inter_size);
    model_config.moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
    model_config.moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 3);
    // For moe group topk config
    model_config.moe_config.num_expert_group = config_json.value("n_group", 1);
    model_config.moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    model_config.moe_config.scoring_func = config_json.value("scoring_func", "sigmoid");
    model_config.moe_config.topk_method = config_json.value("topk_method", "greedy");
    model_config.moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
    model_config.moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.0f);
    if (model_config.moe_config.topk_method == "noaux_tc") {
      model_config.moe_config.use_e_score_correction_bias = true;
    } else {
      model_config.moe_config.use_e_score_correction_bias = false;
    }
  }
  KLLM_LOG_DEBUG << "model_config.moe_config.moe_inter_size " << model_config.moe_config.moe_inter_size;
  model_config.moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (model_config.moe_config.num_shared_experts > 0) {
    model_config.has_shared_experts = true;
    model_config.moe_config.shared_expert_inter_size =
        model_config.moe_config.num_shared_experts * model_config.moe_config.moe_inter_size;
  }
  // For mla config
  model_config.use_mla = true;
  if (config_json.contains("q_lora_rank") && config_json["q_lora_rank"].is_number()) {
    model_config.mla_config.q_lora_rank = config_json.value("q_lora_rank", 1536);
  } else {
    model_config.mla_config.q_lora_rank = 0;
  }
  model_config.mla_config.kv_lora_rank = config_json.value("kv_lora_rank", 512);
  model_config.mla_config.qk_nope_head_dim = config_json.value("qk_nope_head_dim", 128);
  model_config.mla_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  model_config.mla_config.v_head_dim = config_json.value("v_head_dim", 128);

  model_config.size_per_head = model_config.mla_config.qk_nope_head_dim + model_config.mla_config.qk_rope_head_dim;
  KLLM_LOG_INFO << fmt::format(
      "Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, use_mla: {}, "
      "use_e_score_correction_bias: {}",
      model_config.moe_config.num_experts, model_config.moe_config.num_shared_experts,
      model_config.moe_config.experts_topk, model_config.use_mla, model_config.moe_config.use_e_score_correction_bias);
}

}  // namespace ksana_llm
